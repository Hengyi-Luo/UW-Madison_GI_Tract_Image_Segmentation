from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch
from scipy import ndimage

from .constants import CLASSES


DEFAULT_METRIC_NAMES = ("dice", "recall", "precision", "hd95", "rve")


def _validate_shapes(pred: torch.Tensor, target: torch.Tensor) -> None:
    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have the same shape, got {pred.shape} vs {target.shape}")
    if pred.ndim < 3:
        raise ValueError("pred and target must have shape (B, C, ...) for segmentation metrics")


def _to_discrete_mask(x: torch.Tensor, threshold: float) -> torch.Tensor:
    if x.dtype == torch.bool:
        return x
    return x > float(threshold)


def _sample_spacing(
    spacing: Sequence[float] | Sequence[Sequence[float]] | None,
    batch_index: int,
) -> Sequence[float] | None:
    if spacing is None:
        return None
    if len(spacing) == 0:
        return None
    first = spacing[0]
    if isinstance(first, (list, tuple)):
        return spacing[batch_index]  # type: ignore[index]
    return spacing  # type: ignore[return-value]


def _masked_mean_per_class(values: torch.Tensor, valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    num_classes = int(values.shape[1])
    means = torch.full((num_classes,), float("nan"), dtype=torch.float32)
    counts = valid_mask.sum(dim=0).to(torch.int64)
    for class_idx in range(num_classes):
        valid_values = values[:, class_idx][valid_mask[:, class_idx]]
        if valid_values.numel() > 0:
            means[class_idx] = valid_values.float().mean()
    return means, counts


def _macro_mean(values: torch.Tensor) -> float:
    finite_values = values[torch.isfinite(values)]
    if finite_values.numel() == 0:
        return float("nan")
    return float(finite_values.mean().item())


def _normalize_case_ids(case_ids: Sequence[str] | None, batch_size: int, start_index: int) -> list[str]:
    if case_ids is None:
        return [f"case_{start_index + i:06d}" for i in range(batch_size)]
    if len(case_ids) != batch_size:
        raise ValueError(f"Expected {batch_size} case ids, got {len(case_ids)}")
    return [str(x) for x in case_ids]


def _normalize_num_slices(num_slices: Sequence[int] | None, batch_size: int) -> list[int]:
    if num_slices is None:
        return [0 for _ in range(batch_size)]
    if len(num_slices) != batch_size:
        raise ValueError(f"Expected {batch_size} slice counts, got {len(num_slices)}")
    return [int(x) for x in num_slices]


def _surface_distances(
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    *,
    spacing: Sequence[float] | None,
) -> np.ndarray:
    if source_mask.ndim != target_mask.ndim:
        raise ValueError(f"source and target masks must have same ndim, got {source_mask.ndim} vs {target_mask.ndim}")

    structure = ndimage.generate_binary_structure(source_mask.ndim, 1)
    source_eroded = ndimage.binary_erosion(source_mask, structure=structure, border_value=0)
    target_eroded = ndimage.binary_erosion(target_mask, structure=structure, border_value=0)
    source_surface = source_mask ^ source_eroded
    target_surface = target_mask ^ target_eroded

    if not source_surface.any() or not target_surface.any():
        return np.asarray([], dtype=np.float64)

    target_distance = ndimage.distance_transform_edt(~target_surface, sampling=spacing)
    return target_distance[source_surface]


def _hd95_numpy(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    *,
    spacing: Sequence[float] | None,
) -> float:
    forward = _surface_distances(pred_mask, target_mask, spacing=spacing)
    backward = _surface_distances(target_mask, pred_mask, spacing=spacing)
    if forward.size == 0 or backward.size == 0:
        return float("nan")
    return float(max(np.percentile(forward, 95.0), np.percentile(backward, 95.0)))


def _compute_metric_tensors(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    threshold: float,
    spacing: Sequence[float] | Sequence[Sequence[float]] | None,
    metric_names: Sequence[str],
) -> dict[str, torch.Tensor]:
    _validate_shapes(pred, target)

    pred_mask = _to_discrete_mask(pred, threshold=threshold)
    target_mask = _to_discrete_mask(target, threshold=threshold)

    reduce_dims = tuple(range(2, pred.ndim))
    intersection = (pred_mask & target_mask).sum(dim=reduce_dims).float()
    pred_sum = pred_mask.sum(dim=reduce_dims).float()
    target_sum = target_mask.sum(dim=reduce_dims).float()

    gt_present = target_sum > 0
    pred_present = pred_sum > 0
    gt_negative = ~gt_present
    missed = gt_present & (~pred_present)
    spurious = gt_negative & pred_present

    batch_size, num_classes = pred_mask.shape[:2]

    tensors: dict[str, torch.Tensor] = {
        "intersection": intersection,
        "pred_sum": pred_sum,
        "target_sum": target_sum,
        "gt_present": gt_present,
        "gt_negative": gt_negative,
        "pred_present": pred_present,
        "missed": missed,
        "spurious": spurious,
    }

    if "dice" in metric_names:
        dice_valid = gt_present
        dice = torch.full((batch_size, num_classes), float("nan"), dtype=torch.float32, device=pred.device)
        dice[dice_valid] = (2.0 * intersection[dice_valid]) / (pred_sum[dice_valid] + target_sum[dice_valid])
        tensors["dice"] = dice
        tensors["dice_valid"] = dice_valid

    if "recall" in metric_names:
        recall_valid = gt_present
        recall = torch.full((batch_size, num_classes), float("nan"), dtype=torch.float32, device=pred.device)
        recall[recall_valid] = intersection[recall_valid] / target_sum[recall_valid]
        tensors["recall"] = recall
        tensors["recall_valid"] = recall_valid

    if "precision" in metric_names:
        precision_valid = pred_present
        precision = torch.full((batch_size, num_classes), float("nan"), dtype=torch.float32, device=pred.device)
        precision[precision_valid] = intersection[precision_valid] / pred_sum[precision_valid]
        tensors["precision"] = precision
        tensors["precision_valid"] = precision_valid

    if "rve" in metric_names:
        rve_valid = gt_present
        rve = torch.full((batch_size, num_classes), float("nan"), dtype=torch.float32, device=pred.device)
        rve[rve_valid] = (pred_sum[rve_valid] - target_sum[rve_valid]) / target_sum[rve_valid]
        tensors["rve"] = rve
        tensors["rve_valid"] = rve_valid

    if "hd95" in metric_names:
        hd95_valid = gt_present & pred_present
        hd95 = torch.full((batch_size, num_classes), float("nan"), dtype=torch.float32, device=pred.device)
        if bool(hd95_valid.any()):
            for batch_index in range(batch_size):
                sample_spacing = _sample_spacing(spacing, batch_index=batch_index)
                for class_idx in range(num_classes):
                    if not bool(hd95_valid[batch_index, class_idx]):
                        continue
                    pred_np = pred_mask[batch_index, class_idx].detach().cpu().numpy().astype(bool, copy=False)
                    target_np = target_mask[batch_index, class_idx].detach().cpu().numpy().astype(bool, copy=False)
                    value = _hd95_numpy(
                        pred_np,
                        target_np,
                        spacing=tuple(float(x) for x in sample_spacing) if sample_spacing is not None else None,
                    )
                    hd95[batch_index, class_idx] = float(value)
        tensors["hd95"] = hd95
        tensors["hd95_valid"] = hd95_valid

    return tensors


def _concat_metric_tensors(parts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not parts:
        raise ValueError("No metric tensors to concatenate.")
    return {key: torch.cat([part[key] for part in parts], dim=0) for key in parts[0]}


def _build_summary_from_tensors(
    tensors: dict[str, torch.Tensor],
    *,
    class_names: Sequence[str],
    metric_names: Sequence[str],
) -> dict[str, float | int]:
    if not metric_names:
        raise ValueError("metric_names must not be empty")

    first_metric = metric_names[0]
    summary: dict[str, float | int] = {"num_cases": int(tensors[first_metric].shape[0])}

    for metric_name in metric_names:
        means, counts = _masked_mean_per_class(tensors[metric_name], tensors[f"{metric_name}_valid"])
        summary[f"{metric_name}_mean"] = _macro_mean(means)
        if metric_name == "dice":
            summary["macro_dice"] = summary[f"{metric_name}_mean"]
        if metric_name == "recall":
            summary["macro_recall"] = summary[f"{metric_name}_mean"]
        if metric_name == "precision":
            summary["macro_precision"] = summary[f"{metric_name}_mean"]
        if metric_name == "hd95":
            summary["macro_hd95"] = summary[f"{metric_name}_mean"]
        if metric_name == "rve":
            summary["macro_rve"] = summary[f"{metric_name}_mean"]

        for class_idx, class_name in enumerate(class_names):
            summary[f"{metric_name}_{class_name}"] = float(means[class_idx].item())
            summary[f"{metric_name}_valid_n_{class_name}"] = int(counts[class_idx].item())

    gt_positive = tensors["gt_present"].sum(dim=0).to(torch.int64)
    gt_negative = tensors["gt_negative"].sum(dim=0).to(torch.int64)
    pred_positive = tensors["pred_present"].sum(dim=0).to(torch.int64)
    missed = tensors["missed"].sum(dim=0).to(torch.int64)
    spurious = tensors["spurious"].sum(dim=0).to(torch.int64)
    hd95_invalid = (~tensors["hd95_valid"]).sum(dim=0).to(torch.int64) if "hd95" in metric_names else None

    for class_idx, class_name in enumerate(class_names):
        gt_positive_n = int(gt_positive[class_idx].item())
        gt_negative_n = int(gt_negative[class_idx].item())
        pred_positive_n = int(pred_positive[class_idx].item())
        missed_n = int(missed[class_idx].item())
        spurious_n = int(spurious[class_idx].item())
        hd95_invalid_n = int(hd95_invalid[class_idx].item()) if hd95_invalid is not None else 0

        summary[f"gt_positive_n_{class_name}"] = gt_positive_n
        summary[f"gt_negative_n_{class_name}"] = gt_negative_n
        summary[f"pred_positive_n_{class_name}"] = pred_positive_n
        summary[f"missed_organ_n_{class_name}"] = missed_n
        summary[f"spurious_organ_n_{class_name}"] = spurious_n
        summary[f"missed_organ_rate_{class_name}"] = float(missed_n / gt_positive_n) if gt_positive_n > 0 else float("nan")
        summary[f"spurious_organ_rate_{class_name}"] = (
            float(spurious_n / gt_negative_n) if gt_negative_n > 0 else float("nan")
        )
        if hd95_invalid is not None:
            summary[f"hd95_invalid_n_{class_name}"] = hd95_invalid_n

    return summary


def _build_case_rows_from_tensors(
    tensors: dict[str, torch.Tensor],
    *,
    class_names: Sequence[str],
    case_ids: Sequence[str],
    num_slices: Sequence[int],
    metric_names: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not metric_names:
        raise ValueError("metric_names must not be empty")
    num_cases = int(tensors[metric_names[0]].shape[0])

    for batch_index in range(num_cases):
        row: dict[str, Any] = {
            "case_day": str(case_ids[batch_index]),
            "num_slices": int(num_slices[batch_index]),
        }

        for metric_name in metric_names:
            metric_values = tensors[metric_name][batch_index]
            valid_mask = tensors[f"{metric_name}_valid"][batch_index]
            valid_values = metric_values[valid_mask]
            row[f"{metric_name}_mean"] = float(valid_values.mean().item()) if valid_values.numel() > 0 else float("nan")

        row["gt_positive_classes_n"] = int(tensors["gt_present"][batch_index].sum().item())
        row["pred_positive_classes_n"] = int(tensors["pred_present"][batch_index].sum().item())

        for class_idx, class_name in enumerate(class_names):
            for metric_name in metric_names:
                row[f"{metric_name}_{class_name}"] = float(tensors[metric_name][batch_index, class_idx].item())

            row[f"gt_present_{class_name}"] = int(tensors["gt_present"][batch_index, class_idx].item())
            row[f"pred_present_{class_name}"] = int(tensors["pred_present"][batch_index, class_idx].item())
            row[f"missed_organ_{class_name}"] = int(tensors["missed"][batch_index, class_idx].item())
            row[f"spurious_organ_{class_name}"] = int(tensors["spurious"][batch_index, class_idx].item())

            for metric_name in metric_names:
                row[f"{metric_name}_valid_{class_name}"] = int(tensors[f"{metric_name}_valid"][batch_index, class_idx].item())

        rows.append(row)

    return rows


@dataclass
class SegmentationMetricAccumulator:
    class_names: Sequence[str] = field(default_factory=lambda: list(CLASSES))
    metric_names: Sequence[str] = field(default_factory=lambda: list(DEFAULT_METRIC_NAMES))
    threshold: float = 0.5
    _parts: list[dict[str, torch.Tensor]] = field(default_factory=list, init=False)
    _case_ids: list[str] = field(default_factory=list, init=False)
    _num_slices: list[int] = field(default_factory=list, init=False)

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        spacing: Sequence[float] | Sequence[Sequence[float]] | None = None,
        case_ids: Sequence[str] | None = None,
        num_slices: Sequence[int] | None = None,
    ) -> None:
        batch_size = int(pred.shape[0])
        normalized_case_ids = _normalize_case_ids(case_ids, batch_size=batch_size, start_index=len(self._case_ids))
        normalized_num_slices = _normalize_num_slices(num_slices, batch_size=batch_size)

        tensors = _compute_metric_tensors(
            pred,
            target,
            threshold=self.threshold,
            spacing=spacing,
            metric_names=self.metric_names,
        )
        self._parts.append({key: value.detach().cpu() for key, value in tensors.items()})
        self._case_ids.extend(normalized_case_ids)
        self._num_slices.extend(normalized_num_slices)

    def summary(self) -> dict[str, float | int]:
        if not self._parts:
            return {"num_cases": 0}
        tensors = _concat_metric_tensors(self._parts)
        return _build_summary_from_tensors(tensors, class_names=self.class_names, metric_names=self.metric_names)

    def per_case_rows(self) -> list[dict[str, Any]]:
        if not self._parts:
            return []
        tensors = _concat_metric_tensors(self._parts)
        return _build_case_rows_from_tensors(
            tensors,
            class_names=self.class_names,
            case_ids=self._case_ids,
            num_slices=self._num_slices,
            metric_names=self.metric_names,
        )


def dice_per_class(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    tensors = _compute_metric_tensors(pred, target, threshold=threshold, spacing=None, metric_names=["dice"])
    means, _ = _masked_mean_per_class(tensors["dice"], tensors["dice_valid"])
    return means


def precision_per_class(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    tensors = _compute_metric_tensors(pred, target, threshold=threshold, spacing=None, metric_names=["precision"])
    means, _ = _masked_mean_per_class(tensors["precision"], tensors["precision_valid"])
    return means


def recall_per_class(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    tensors = _compute_metric_tensors(pred, target, threshold=threshold, spacing=None, metric_names=["recall"])
    means, _ = _masked_mean_per_class(tensors["recall"], tensors["recall_valid"])
    return means


def hd95_per_class(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    spacing: Sequence[float] | Sequence[Sequence[float]] | None = None,
) -> torch.Tensor:
    tensors = _compute_metric_tensors(pred, target, threshold=threshold, spacing=spacing, metric_names=["hd95"])
    means, _ = _masked_mean_per_class(tensors["hd95"], tensors["hd95_valid"])
    return means


def rve_per_class(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    tensors = _compute_metric_tensors(pred, target, threshold=threshold, spacing=None, metric_names=["rve"])
    means, _ = _masked_mean_per_class(tensors["rve"], tensors["rve_valid"])
    return means


def summarize_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    class_names: Sequence[str] | None = None,
    threshold: float = 0.5,
    spacing: Sequence[float] | Sequence[Sequence[float]] | None = None,
) -> dict[str, float | int]:
    class_names = list(class_names or CLASSES)
    if pred.shape[1] != len(class_names):
        raise ValueError(f"expected {len(class_names)} classes, got channel dimension {pred.shape[1]}")

    tensors = _compute_metric_tensors(
        pred,
        target,
        threshold=threshold,
        spacing=spacing,
        metric_names=DEFAULT_METRIC_NAMES,
    )
    return _build_summary_from_tensors(tensors, class_names=class_names, metric_names=DEFAULT_METRIC_NAMES)
