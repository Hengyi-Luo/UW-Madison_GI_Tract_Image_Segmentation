#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from scripts.nnunet_common import (  # noqa: E402
    dataclass_from_raw,
    dataset_folder_name,
    extract_python_cfg,
    json_safe,
    load_case_day_split,
    load_python_module,
    parse_nnunet_validation_summary,
    setup_logger,
)
from src.constants import CLASSES  # noqa: E402
from src.data_utils import build_case_day_slices, build_rle_index  # noqa: E402
from src.metrics import SegmentationMetricAccumulator  # noqa: E402
from src.rle import rle_decode, rle_encode  # noqa: E402


@dataclass(frozen=True)
class EvalNNUNetCfg:
    data_root: str = "inputs"
    train_csv: str = "inputs/train.csv"
    val_ids: str = "inputs/splits/val_case_days.csv"

    dataset_id: int = 501
    dataset_name: str = "UWGI"
    nnunet_results_dir: str = "nnUNet_results"
    trainer: str = "nnUNetTrainer"
    plans: str = "nnUNetResEncUNetPlans"
    configuration: str = "3d_fullres"
    fold: int = 0

    threshold: float = 0.5
    output_dir: str = ""


def _load_eval_cfg(config_path: str) -> EvalNNUNetCfg:
    module = load_python_module(Path(config_path), module_name="eval_nnunet_cfg")
    raw = extract_python_cfg(module, Path(config_path))
    filtered = {k: v for k, v in raw.items() if k in {f.name for f in EvalNNUNetCfg.__dataclass_fields__.values()}}
    return dataclass_from_raw(EvalNNUNetCfg, filtered)


def _results_case_dir(cfg: EvalNNUNetCfg) -> Path:
    dataset_dir = dataset_folder_name(cfg.dataset_id, cfg.dataset_name)
    return (
        REPO_ROOT
        / cfg.nnunet_results_dir
        / dataset_dir
        / f"{cfg.trainer}__{cfg.plans}__{cfg.configuration}"
        / f"fold_{int(cfg.fold)}"
    )


def _read_pred_label(path: Path) -> np.ndarray:
    data = np.asarray(nib.load(str(path)).dataobj)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D label map for {path}, got shape {data.shape}")
    pred = np.transpose(data, (2, 1, 0))
    return np.rint(pred).astype(np.uint8)


def _build_gt_multilabel(
    *,
    case_day: str,
    slice_paths: list[str],
    rle_index: dict[tuple[str, int], dict[str, str]],
) -> np.ndarray:
    if not slice_paths:
        raise ValueError(f"Missing slice paths for {case_day}")
    parts = Path(slice_paths[0]).stem.split("_")
    width = int(parts[2])
    height = int(parts[3])
    gt = np.zeros((len(CLASSES), len(slice_paths), height, width), dtype=np.uint8)
    for depth, slice_path in enumerate(slice_paths):
        slice_idx = int(Path(slice_path).stem.split("_")[1])
        per_class = rle_index.get((case_day, slice_idx), {})
        for class_index, class_name in enumerate(CLASSES):
            gt[class_index, depth] = rle_decode(per_class.get(class_name, ""), height, width).astype(np.uint8)
    return gt


def _pred_to_multilabel(pred_zyx: np.ndarray) -> np.ndarray:
    return np.stack([(pred_zyx == (class_index + 1)).astype(np.uint8) for class_index in range(len(CLASSES))], axis=0)


def evaluate_nnunet_validation(cfg: Any, *, out_dir: Path, logger=None) -> dict[str, Any]:
    eval_cfg = EvalNNUNetCfg(
        data_root=str(getattr(cfg, "data_root")),
        train_csv=str(getattr(cfg, "train_csv")),
        val_ids=str(getattr(cfg, "val_ids")),
        dataset_id=int(getattr(cfg, "dataset_id")),
        dataset_name=str(getattr(cfg, "dataset_name")),
        nnunet_results_dir=str(getattr(cfg, "nnunet_results_dir")),
        trainer=str(getattr(cfg, "trainer", "nnUNetTrainer")),
        plans=str(getattr(cfg, "plans", "nnUNetResEncUNetPlans")),
        configuration=str(getattr(cfg, "configuration", "3d_fullres")),
        fold=int(getattr(cfg, "fold", 0)),
        threshold=float(getattr(cfg, "threshold", 0.5)),
        output_dir=str(out_dir),
    )

    result_case_dir = _results_case_dir(eval_cfg)
    validation_dir = result_case_dir / "validation"
    validation_summary_path = validation_dir / "summary.json"
    if not validation_dir.is_dir():
        raise FileNotFoundError(f"nnUNet validation directory not found: {validation_dir}")
    if not validation_summary_path.exists():
        raise FileNotFoundError(f"nnUNet validation summary not found: {validation_summary_path}")

    train_dir = REPO_ROOT / eval_cfg.data_root / "train"
    case_day_slices = build_case_day_slices(str(train_dir))
    rle_index = build_rle_index(str(REPO_ROOT / eval_cfg.train_csv))
    val_case_days = load_case_day_split(eval_cfg.val_ids)
    metric_acc = SegmentationMetricAccumulator(class_names=CLASSES, threshold=float(eval_cfg.threshold))
    submit_rows: list[dict[str, str]] = []

    for case_day in val_case_days:
        pred_path = validation_dir / f"{case_day}.nii.gz"
        if not pred_path.exists():
            raise FileNotFoundError(f"Validation prediction missing for {case_day}: {pred_path}")
        pred_zyx = _read_pred_label(pred_path)
        slice_paths = case_day_slices.get(case_day)
        if not slice_paths:
            raise KeyError(f"case_day not found in source data: {case_day}")
        gt_cdhw = _build_gt_multilabel(case_day=case_day, slice_paths=slice_paths, rle_index=rle_index)
        pred_cdhw = _pred_to_multilabel(pred_zyx)
        if pred_cdhw.shape[1:] != gt_cdhw.shape[1:]:
            raise ValueError(
                f"Prediction shape mismatch for {case_day}: pred={pred_cdhw.shape} gt={gt_cdhw.shape}"
            )

        metric_acc.update(
            torch.from_numpy(pred_cdhw[None]).float(),
            torch.from_numpy(gt_cdhw[None]).float(),
            case_ids=[case_day],
            num_slices=[int(gt_cdhw.shape[1])],
        )

        for depth, slice_path in enumerate(slice_paths):
            slice_idx = int(Path(slice_path).stem.split("_")[1])
            row_id = f"{case_day}_slice_{slice_idx:04d}"
            for class_index, class_name in enumerate(CLASSES):
                submit_rows.append(
                    {
                        "id": row_id,
                        "class": class_name,
                        "segmentation": rle_encode(pred_cdhw[class_index, depth]),
                    }
                )

        if logger is not None:
            logger.info("Evaluated validation prediction for %s", case_day)

    out_dir.mkdir(parents=True, exist_ok=True)
    official_metrics = parse_nnunet_validation_summary(
        validation_summary_path,
        class_names=list(CLASSES),
        label_values=[1, 2, 3],
    )
    per_case_rows = metric_acc.per_case_rows()
    eval_summary_metrics = metric_acc.summary()

    eval_per_case_path = out_dir / "eval_per_case.csv"
    with open(eval_per_case_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = list(per_case_rows[0].keys()) if per_case_rows else ["case_day", "num_slices", "dice_mean"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_case_rows)

    eval_summary = {
        "protocol": "docs/METRICS_PROTOCOL.md",
        "validation_dir": str(validation_dir),
        "validation_summary_json": str(validation_summary_path),
        "num_case_days": int(len(per_case_rows)),
        **official_metrics,
        **eval_summary_metrics,
        "mean_dice": float(eval_summary_metrics["dice_mean"]),
        "mean_dice_per_class": [float(eval_summary_metrics[f"dice_{class_name}"]) for class_name in CLASSES],
        "threshold": float(eval_cfg.threshold),
    }
    eval_summary_path = out_dir / "eval_summary.json"
    eval_summary_path.write_text(json.dumps(json_safe(eval_summary), indent=2) + "\n", encoding="utf-8")

    submit_csv_path = out_dir / "submit.csv"
    with open(submit_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "class", "segmentation"])
        writer.writeheader()
        writer.writerows(submit_rows)

    return {
        "result_case_dir": result_case_dir,
        "validation_dir": validation_dir,
        "validation_summary_path": validation_summary_path,
        "official_metrics": official_metrics,
        "eval_summary_metrics": eval_summary_metrics,
        "eval_summary_path": eval_summary_path,
        "eval_per_case_path": eval_per_case_path,
        "submit_csv_path": submit_csv_path,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate nnUNet validation predictions with repo metrics.")
    parser.add_argument("--config", type=str, default="configs/train_nnunet_baseline.py")
    parser.add_argument("--output-dir", type=str, default="", help="Override output directory for eval artifacts")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    cfg = _load_eval_cfg(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else (REPO_ROOT / "outputs" / "nnunet_eval")
    logger = setup_logger(output_dir / "eval_nnunet_baseline.log", name="eval_nnunet_baseline")
    result = evaluate_nnunet_validation(cfg, out_dir=output_dir, logger=logger)
    logger.info("Official validation summary: %s", result["validation_summary_path"])
    logger.info("Wrote eval summary         : %s", result["eval_summary_path"])
    logger.info("Wrote eval per-case        : %s", result["eval_per_case_path"])
    logger.info("Wrote submit.csv           : %s", result["submit_csv_path"])


if __name__ == "__main__":
    main()
