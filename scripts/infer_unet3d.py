#!/usr/bin/env python3
"""
Inference script: load a trained 3D U-Net checkpoint, run full-volume inference on a split of case_days,
and write a Kaggle-style RLE CSV (same schema as inputs/train.csv) named submit.csv under the model's
outputs directory.

Also optionally computes case_day-level Dice evaluation and summary (when `evaluate: true`) by decoding
GT masks from inputs/train.csv.

Usage:
  python scripts/infer_unet3d.py --config configs/infer_unet3d_benchmark.py

Default output:
  <weights_parent>/infer/submit.csv
  <weights_parent>/infer/eval_per_case.csv
  <weights_parent>/infer/eval_summary.json
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import argparse
import csv
import importlib.util
import json
import math
import os
import sys
from typing import Any

import nibabel as nib
import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.networks.layers import Norm
from monai.networks.nets import Unet
from monai.transforms import Compose, EnsureChannelFirstd, EnsureTyped, Lambdad, ScaleIntensityd, SpatialPadd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)


from src.constants import CLASSES  # noqa: E402
from src.data_utils import build_case_day_slices, build_rle_index, load_case_days  # noqa: E402
from src.datasets import LoadCaseDayImaged, LoadCaseDayVolumed  # noqa: E402
from src.metrics import SegmentationMetricAccumulator  # noqa: E402
from src.rle import rle_encode  # noqa: E402


@dataclass(frozen=True)
class InferCfg:
    data_root: str = "inputs"
    ids_csv: str = "inputs/splits/val_case_days.csv"
    train_csv: str = "inputs/train.csv"

    weights: str | list[str] = ""
    output_dir: str = ""  # default: <weights_parent>/infer

    out_csv_name: str = "submit.csv"
    export_pred_nifti: bool = False
    pred_nifti_dir_name: str = "pred_nifti"

    evaluate: bool = True
    eval_per_case_name: str = "eval_per_case.csv"
    eval_summary_name: str = "eval_summary.json"

    roi_d: int = 96
    roi_h: int = 224
    roi_w: int = 224
    sw_batch_size: int = 4
    overlap: float = 0.25
    sw_mode: str = "constant"  # constant | gaussian
    sw_sigma_scale: float = 0.125
    threshold: float = 0.5
    tta_flips: list[list[int]] = field(default_factory=list)  # spatial axes relative to (D,H,W)
    intensity_norm: str = "scale_intensity"
    pad_to_roi: bool = True

    num_workers: int = 0
    device: str = ""

    unsafe_load: bool = False

    @property
    def roi_size(self) -> tuple[int, int, int]:
        return (int(self.roi_d), int(self.roi_h), int(self.roi_w))

    @property
    def eval_enabled(self) -> bool:
        return bool(self.evaluate)

    @property
    def weight_paths(self) -> list[str]:
        if isinstance(self.weights, str):
            return [self.weights] if self.weights else []
        return [str(x) for x in self.weights if str(x).strip()]


def _normalize_weight_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                decoded = json.loads(text)
            except Exception:
                decoded = None
            if isinstance(decoded, list):
                return [str(x).strip() for x in decoded if str(x).strip()]
        if "," in text:
            return [part.strip() for part in text.split(",") if part.strip()]
        return [text]
    return [str(value).strip()]


def _normalize_tta_flips(value: Any) -> list[list[int]]:
    if value in (None, "", []):
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        decoded = json.loads(text)
        value = decoded
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"tta_flips must be a list of spatial-axis lists, got {type(value).__name__}")
    flips: list[list[int]] = []
    for axes in value:
        if isinstance(axes, (list, tuple)):
            flip_axes = [int(axis) for axis in axes]
            if flip_axes:
                flips.append(flip_axes)
        else:
            flips.append([int(axes)])
    return flips


def _normalize_infer_cfg_raw(raw: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(raw)
    weight_list = _normalize_weight_list(normalized.get("weights", ""))
    normalized["weights"] = weight_list[0] if len(weight_list) == 1 else weight_list
    normalized["tta_flips"] = _normalize_tta_flips(normalized.get("tta_flips", []))
    normalized["sw_mode"] = str(normalized.get("sw_mode", "constant")).strip().lower()
    return normalized


def _load_python_module(path: Path, *, module_name: str):
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Python module not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load Python module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _extract_python_cfg(module: Any, path: Path) -> dict[str, Any]:
    cfg_obj = getattr(module, "cfg", None)
    if cfg_obj is None:
        raise AttributeError(f"Python config '{path}' must define a top-level 'cfg'")
    if isinstance(cfg_obj, dict):
        return dict(cfg_obj)
    if hasattr(cfg_obj, "__dict__"):
        return {k: v for k, v in vars(cfg_obj).items() if not k.startswith("_")}
    raise TypeError(f"Unsupported cfg type in '{path}': {type(cfg_obj).__name__}")


def _load_cfg(path: str) -> InferCfg:
    cfg_path = Path(path)
    suffix = cfg_path.suffix.lower()
    if suffix != ".py":
        raise ValueError(f"Config must be a Python module (.py), got: {cfg_path}")
    module = _load_python_module(cfg_path, module_name="infer_unet3d_config")
    raw = _extract_python_cfg(module, cfg_path)

    known = set(InferCfg().__dict__.keys())
    unknown = sorted([k for k in raw.keys() if k not in known])
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")

    merged = {**InferCfg().__dict__, **raw}
    merged = _normalize_infer_cfg_raw(merged)
    return InferCfg(**merged)


def _apply_cfg_overrides(cfg: InferCfg, **overrides: Any) -> InferCfg:
    raw = {**asdict(cfg), **overrides}
    raw = _normalize_infer_cfg_raw(raw)
    return InferCfg(**raw)


def _build_model(device: torch.device) -> torch.nn.Module:
    return Unet(
        spatial_dims=3,
        in_channels=1,
        out_channels=len(CLASSES),
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=0.2,
        norm=Norm.BATCH,
    ).to(device)


def _safe_unit_scale(x: np.ndarray) -> np.ndarray:
    max_value = float(np.max(x))
    if max_value <= 0.0:
        return x.astype(np.float32)
    return (x / max_value).astype(np.float32)


def _get_intensity_transform(norm_name: str):
    norm_name = str(norm_name).strip().lower()
    if norm_name == "scale_intensity":
        return ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0)
    if norm_name == "safe_unit_scale":
        return Lambdad(keys=["image"], func=_safe_unit_scale)
    raise ValueError(f"Unsupported intensity_norm: {norm_name}")


def _get_infer_transforms(*, cfg: InferCfg, case_day_slices):
    transforms: list[Any] = [
        LoadCaseDayImaged(keys=["case_day"], case_day_slices=case_day_slices),
        EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        _get_intensity_transform(cfg.intensity_norm),
    ]
    if bool(cfg.pad_to_roi):
        transforms.append(
            SpatialPadd(
                keys=["image"],
                spatial_size=(int(cfg.roi_d), int(cfg.roi_h), int(cfg.roi_w)),
                mode="constant",
            )
        )
    transforms.append(EnsureTyped(keys=["image"], dtype=torch.float32))
    return Compose(transforms)


def _get_eval_transforms(*, cfg: InferCfg, case_day_slices, rle_index):
    transforms: list[Any] = [
        LoadCaseDayVolumed(keys=["case_day"], case_day_slices=case_day_slices, rle_index=rle_index),
        EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        _get_intensity_transform(cfg.intensity_norm),
    ]
    if bool(cfg.pad_to_roi):
        transforms.append(
            SpatialPadd(
                keys=["image", "label"],
                spatial_size=(int(cfg.roi_d), int(cfg.roi_h), int(cfg.roi_w)),
                mode="constant",
            )
        )
    transforms.append(EnsureTyped(keys=["image", "label"], dtype=torch.float32))
    return Compose(transforms)


def _load_checkpoint(weights_path: str, device: torch.device, *, unsafe_load: bool) -> dict[str, Any]:
    try:
        ckpt = torch.load(weights_path, map_location=device, weights_only=not bool(unsafe_load))
    except Exception as e:
        if "Weights only load failed" in str(e):
            import numpy as _np

            try:
                from torch.serialization import safe_globals
            except Exception:
                safe_globals = None

            if safe_globals is not None:
                with safe_globals([_np._core.multiarray._reconstruct]):
                    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
            else:
                raise
        else:
            raise
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint must be a dict.")
    return ckpt


def _extract_state_dict(ckpt: dict[str, Any]) -> dict[str, Any]:
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        raise ValueError("Unrecognized checkpoint format: expected model_state_dict or model keys.")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint model payload must be a state dict.")
    return state_dict


def _resolve_output_dir(cfg: InferCfg) -> Path:
    if cfg.output_dir:
        return Path(cfg.output_dir)
    weight_paths = cfg.weight_paths
    if not weight_paths:
        raise ValueError("weights must be set (either in config or via --weights).")
    base_dir = Path(weight_paths[0]).resolve().parent
    if len(weight_paths) == 1:
        return base_dir / "infer"
    return base_dir / f"infer_ensemble_{len(weight_paths)}"


def _write_submit_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "class", "segmentation"])
        w.writeheader()
        w.writerows(rows)


def _parse_scan_filename(filename: str) -> tuple[int, int, int, float, float]:
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 6:
        raise ValueError(f"Unexpected scan filename: {filename}")
    return int(parts[1]), int(parts[3]), int(parts[2]), float(parts[4]), float(parts[5])


def _save_nifti(path: Path, volume_zyx: np.ndarray, spacing_xyz: tuple[float, float, float]) -> None:
    affine = np.diag([spacing_xyz[0], spacing_xyz[1], spacing_xyz[2], 1.0]).astype(np.float32)
    image = nib.Nifti1Image(np.transpose(volume_zyx, (2, 1, 0)), affine)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(path))


def _export_pred_masks_nifti(
    *,
    out_dir: Path,
    dir_name: str,
    case_day: str,
    pred_cdhw: np.ndarray,
    case_day_slices: dict[str, list[str]],
) -> None:
    slice_paths = case_day_slices.get(case_day)
    if not slice_paths:
        raise KeyError(f"Missing slice paths for case_day: {case_day}")
    _, _, _, spacing_x, spacing_y = _parse_scan_filename(Path(slice_paths[0]).name)
    spacing = (float(spacing_x), float(spacing_y), float(spacing_x))

    case_dir = out_dir / str(dir_name) / case_day
    for ci, cls_name in enumerate(CLASSES):
        mask_path = case_dir / f"{case_day}_mask_{cls_name}.nii.gz"
        _save_nifti(mask_path, pred_cdhw[ci].astype(np.uint8), spacing)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _spatial_axes_to_tensor_dims(spatial_axes: list[int]) -> list[int]:
    dims: list[int] = []
    for axis in spatial_axes:
        axis = int(axis)
        if axis < 0 or axis > 2:
            raise ValueError(f"tta spatial axis must be one of 0, 1, 2 for (D,H,W), got: {axis}")
        dims.append(axis + 2)
    return dims


def _sliding_window_logits(
    *,
    model: torch.nn.Module,
    x: torch.Tensor,
    roi_size: tuple[int, int, int],
    sw_batch_size: int,
    overlap: float,
    sw_mode: str,
    sw_sigma_scale: float,
) -> torch.Tensor:
    kwargs: dict[str, Any] = {
        "roi_size": roi_size,
        "sw_batch_size": int(sw_batch_size),
        "predictor": model,
        "overlap": float(overlap),
    }
    if str(sw_mode).strip().lower() == "gaussian":
        kwargs["mode"] = "gaussian"
        kwargs["sigma_scale"] = float(sw_sigma_scale)
    else:
        kwargs["mode"] = "constant"
    return sliding_window_inference(x, **kwargs)


def _predict_probs(
    *,
    model: torch.nn.Module,
    state_dicts: list[dict[str, Any]],
    x: torch.Tensor,
    roi_size: tuple[int, int, int],
    sw_batch_size: int,
    overlap: float,
    sw_mode: str,
    sw_sigma_scale: float,
    tta_flips: list[list[int]],
) -> torch.Tensor:
    probs_sum: torch.Tensor | None = None
    num_predictions = 0
    for state_dict in state_dicts:
        model.load_state_dict(state_dict)
        logits = _sliding_window_logits(
            model=model,
            x=x,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            overlap=overlap,
            sw_mode=sw_mode,
            sw_sigma_scale=sw_sigma_scale,
        )
        probs = torch.sigmoid(logits)
        probs_sum = probs if probs_sum is None else probs_sum + probs
        num_predictions += 1

        for spatial_axes in tta_flips:
            flip_dims = _spatial_axes_to_tensor_dims(spatial_axes)
            flip_logits = _sliding_window_logits(
                model=model,
                x=torch.flip(x, dims=flip_dims),
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                overlap=overlap,
                sw_mode=sw_mode,
                sw_sigma_scale=sw_sigma_scale,
            )
            flip_probs = torch.sigmoid(flip_logits)
            flip_probs = torch.flip(flip_probs, dims=flip_dims)
            probs_sum = flip_probs if probs_sum is None else probs_sum + flip_probs
            num_predictions += 1

    if probs_sum is None or num_predictions <= 0:
        raise RuntimeError("No predictions were produced during inference.")
    return probs_sum / float(num_predictions)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/infer_unet3d_benchmark.py")
    p.add_argument("--weights", type=str, default="", help="Override cfg.weights")
    p.add_argument("--ids_csv", type=str, default="", help="Override cfg.ids_csv")
    p.add_argument("--output_dir", type=str, default="", help="Override cfg.output_dir")
    p.add_argument("--device", type=str, default="", help="Override cfg.device")
    p.add_argument("--num_workers", type=int, default=None, help="Override cfg.num_workers")
    p.add_argument(
        "--tta_flips",
        type=str,
        default="",
        help="Override cfg.tta_flips with JSON, for example '[[1],[2],[1,2]]' or '[]'.",
    )
    p.add_argument(
        "--disable_tta",
        action="store_true",
        help="Force cfg.tta_flips=[] regardless of config defaults.",
    )
    p.add_argument(
        "--export-pred-nifti",
        action="store_true",
        help="Also export one predicted mask NIfTI per class and case_day.",
    )
    p.add_argument(
        "--unsafe_load",
        action="store_true",
        help="Set weights_only=False when loading checkpoint (unsafe unless you trust the source).",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    cfg = _load_cfg(args.config) if args.config and os.path.exists(args.config) else InferCfg()
    if args.weights:
        cfg = _apply_cfg_overrides(cfg, weights=args.weights)
    if args.ids_csv:
        cfg = _apply_cfg_overrides(cfg, ids_csv=args.ids_csv)
    if args.output_dir:
        cfg = _apply_cfg_overrides(cfg, output_dir=args.output_dir)
    if args.device:
        cfg = _apply_cfg_overrides(cfg, device=args.device)
    if args.num_workers is not None:
        cfg = _apply_cfg_overrides(cfg, num_workers=int(args.num_workers))
    if args.tta_flips:
        cfg = _apply_cfg_overrides(cfg, tta_flips=args.tta_flips)
    if args.disable_tta:
        cfg = _apply_cfg_overrides(cfg, tta_flips=[])
    if args.export_pred_nifti:
        cfg = _apply_cfg_overrides(cfg, export_pred_nifti=True)
    if args.unsafe_load:
        cfg = _apply_cfg_overrides(cfg, unsafe_load=True)

    weight_paths = cfg.weight_paths
    if not weight_paths:
        raise ValueError("Missing weights. Set it in config or pass --weights.")
    for weight_path in weight_paths:
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Weights not found: {weight_path}")
    if not os.path.exists(cfg.ids_csv):
        raise FileNotFoundError(f"ids_csv not found: {cfg.ids_csv}")

    out_dir = _resolve_output_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "config.resolved.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n", encoding="utf-8")
    if args.config and os.path.exists(args.config):
        (out_dir / "config.py").write_text(Path(args.config).read_text(encoding="utf-8"), encoding="utf-8")

    device = torch.device(cfg.device) if cfg.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_root = cfg.data_root
    train_dir = os.path.join(data_root, "train")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    print("Building slice index...")
    case_day_slices = build_case_day_slices(train_dir)
    case_days = load_case_days(cfg.ids_csv)
    case_days = [cd for cd in case_days if cd in case_day_slices]
    if not case_days:
        raise ValueError("No valid case_days for inference after filtering.")
    print(f"Case_days: {len(case_days)}")

    infer_files = [{"case_day": cd} for cd in case_days]
    metric_acc = SegmentationMetricAccumulator(class_names=CLASSES, threshold=float(cfg.threshold))

    if cfg.eval_enabled:
        if not os.path.exists(cfg.train_csv):
            raise FileNotFoundError(f"train_csv not found (needed for evaluation): {cfg.train_csv}")
        print("Building RLE index (for evaluation)...")
        rle_index = build_rle_index(cfg.train_csv)
        infer_transforms = _get_eval_transforms(
            cfg=cfg,
            case_day_slices=case_day_slices,
            rle_index=rle_index,
        )
    else:
        infer_transforms = _get_infer_transforms(
            cfg=cfg,
            case_day_slices=case_day_slices,
        )

    infer_ds = Dataset(data=infer_files, transform=infer_transforms)

    def _make_loader(num_workers: int) -> DataLoader:
        return DataLoader(
            infer_ds,
            batch_size=1,
            shuffle=False,
            num_workers=int(num_workers),
            pin_memory=(device.type == "cuda"),
        )

    infer_loader = _make_loader(int(cfg.num_workers))
    try:
        _ = iter(infer_loader)
    except PermissionError:
        if int(cfg.num_workers) != 0:
            print("[WARN] DataLoader multiprocessing not permitted; retrying with num_workers=0")
            infer_loader = _make_loader(0)

    model = _build_model(device)
    state_dicts = [
        _extract_state_dict(_load_checkpoint(weight_path, device, unsafe_load=bool(cfg.unsafe_load)))
        for weight_path in weight_paths
    ]
    model.eval()

    roi_size = cfg.roi_size
    sw_batch_size = int(cfg.sw_batch_size)
    overlap = float(cfg.overlap)
    thr = float(cfg.threshold)
    sw_mode = str(cfg.sw_mode).strip().lower()
    sw_sigma_scale = float(cfg.sw_sigma_scale)

    rows: list[dict[str, str]] = []
    with torch.no_grad():
        for batch in infer_loader:
            x = batch["image"].to(device)
            case_day = str(batch["case_day"][0])
            orig_d, orig_h, orig_w = [int(t.item()) for t in batch["orig_shape"]]
            slice_idxs = [int(t.item()) for t in batch["slice_idxs"]]

            probs = _predict_probs(
                model=model,
                state_dicts=state_dicts,
                x=x,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                overlap=overlap,
                sw_mode=sw_mode,
                sw_sigma_scale=sw_sigma_scale,
                tta_flips=cfg.tta_flips,
            )
            pred = (probs > thr).to(torch.uint8)  # (1,C,D,H,W)

            pred_cdhw = pred[0, :, :orig_d, :orig_h, :orig_w].detach().cpu().numpy()

            if bool(cfg.export_pred_nifti):
                _export_pred_masks_nifti(
                    out_dir=out_dir,
                    dir_name=str(cfg.pred_nifti_dir_name),
                    case_day=case_day,
                    pred_cdhw=pred_cdhw,
                    case_day_slices=case_day_slices,
                )

            if cfg.eval_enabled:
                y = batch["label"].to(device).to(torch.uint8)  # (1,C,D,H,W)
                metric_acc.update(
                    pred[:, :, :orig_d, :orig_h, :orig_w].float(),
                    y[:, :, :orig_d, :orig_h, :orig_w].float(),
                    case_ids=[case_day],
                    num_slices=[int(orig_d)],
                )

            for z in range(int(orig_d)):
                slice_idx = int(slice_idxs[z]) if z < len(slice_idxs) else (z + 1)
                _id = f"{case_day}_slice_{slice_idx:04d}"
                for ci, cls in enumerate(CLASSES):
                    m2d = pred_cdhw[ci, z]
                    seg = rle_encode(m2d)
                    rows.append({"id": _id, "class": str(cls), "segmentation": str(seg)})

    out_csv = out_dir / str(cfg.out_csv_name)
    _write_submit_csv(out_csv, rows)

    if cfg.eval_enabled:
        per_case_rows = metric_acc.per_case_rows()
        eval_summary_metrics = metric_acc.summary()
        eval_per_case_path = out_dir / str(cfg.eval_per_case_name)
        with open(eval_per_case_path, "w", encoding="utf-8", newline="") as f:
            fieldnames = list(per_case_rows[0].keys()) if per_case_rows else ["case_day", "num_slices", "dice_mean"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(per_case_rows)

        eval_summary = {
            "data_root": str(Path(cfg.data_root).resolve()),
            "train_csv": str(Path(cfg.train_csv).resolve()),
            "weights": [str(Path(weight).resolve()) for weight in weight_paths],
            "ids_csv": str(Path(cfg.ids_csv).resolve()),
            "protocol": "docs/METRICS_PROTOCOL.md",
            "hd95_units": "voxels",
            "num_case_days": int(len(per_case_rows)),
            **eval_summary_metrics,
            "mean_dice": float(eval_summary_metrics["dice_mean"]),
            "mean_dice_per_class": [float(eval_summary_metrics[f"dice_{cls_name}"]) for cls_name in CLASSES],
            "threshold": float(thr),
            "roi_size": list(roi_size),
            "sw_batch_size": int(sw_batch_size),
            "overlap": float(overlap),
            "sw_mode": str(sw_mode),
            "sw_sigma_scale": float(sw_sigma_scale),
            "tta_flips": [list(x) for x in cfg.tta_flips],
            "num_models": int(len(weight_paths)),
            "num_predictions_averaged": int(len(weight_paths) * (1 + len(cfg.tta_flips))),
        }
        (out_dir / str(cfg.eval_summary_name)).write_text(
            json.dumps(_json_safe(eval_summary), indent=2) + "\n",
            encoding="utf-8",
        )

    summary = {
        "weights": [str(Path(weight).resolve()) for weight in weight_paths],
        "ids_csv": str(Path(cfg.ids_csv).resolve()),
        "num_case_days": int(len(case_days)),
        "num_rows": int(len(rows)),
        "threshold": float(thr),
        "roi_size": list(roi_size),
        "sw_batch_size": int(sw_batch_size),
        "overlap": float(overlap),
        "sw_mode": str(sw_mode),
        "sw_sigma_scale": float(sw_sigma_scale),
        "tta_flips": [list(x) for x in cfg.tta_flips],
        "num_models": int(len(weight_paths)),
        "num_predictions_averaged": int(len(weight_paths) * (1 + len(cfg.tta_flips))),
        "output_csv": str(out_csv),
        "export_pred_nifti": bool(cfg.export_pred_nifti),
        "pred_nifti_dir": str(out_dir / str(cfg.pred_nifti_dir_name)),
    }
    (out_dir / "infer_summary.json").write_text(json.dumps(_json_safe(summary), indent=2) + "\n", encoding="utf-8")

    print(f"Wrote submission CSV: {out_csv}")
    if bool(cfg.export_pred_nifti):
        print(f"Wrote predicted masks: {out_dir / str(cfg.pred_nifti_dir_name)}")
    if cfg.eval_enabled:
        print(f"Wrote eval per-case : {out_dir / str(cfg.eval_per_case_name)}")
        print(f"Wrote eval summary  : {out_dir / str(cfg.eval_summary_name)}")
    print(f"Wrote summary        : {out_dir / 'infer_summary.json'}")


if __name__ == "__main__":
    main()
