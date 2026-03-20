#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from scripts.nnunet_common import (  # noqa: E402
    cfg_to_serializable_dict,
    dataclass_from_raw,
    dataset_folder_name,
    extract_python_cfg,
    json_safe,
    load_case_day_split,
    load_python_module,
    setup_logger,
)
from src.constants import CLASSES  # noqa: E402
from src.data_utils import build_case_day_slices, build_rle_index  # noqa: E402
from src.rle import rle_decode  # noqa: E402


@dataclass(frozen=True)
class PrepareNNUNetCfg:
    data_root: str = "inputs"
    train_csv: str = "inputs/train.csv"
    train_ids: str = "inputs/splits/train_case_days.csv"
    val_ids: str = "inputs/splits/val_case_days.csv"

    dataset_id: int = 501
    dataset_name: str = "UWGI"
    nnunet_raw_dir: str = "nnUNet_raw"

    label_collapse_mode: str = "priority_overwrite"
    label_priority: list[str] = field(default_factory=lambda: list(CLASSES))

    spacing_z: float | None = None
    overwrite: bool = False


def _parse_scan_filename(filename: str) -> tuple[int, int, int, float, float]:
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 6:
        raise ValueError(f"Unexpected scan filename: {filename}")
    slice_idx = int(parts[1])
    width = int(parts[2])
    height = int(parts[3])
    spacing_x = float(parts[4])
    spacing_y = float(parts[5])
    return slice_idx, height, width, spacing_x, spacing_y


def _read_scan(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to read scan: {path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _save_nifti(path: Path, volume_zyx: np.ndarray, *, spacing_xyz: tuple[float, float, float], description: str) -> None:
    affine = np.diag([spacing_xyz[0], spacing_xyz[1], spacing_xyz[2], 1.0]).astype(np.float32)
    image_xyz = np.transpose(volume_zyx, (2, 1, 0))
    nii = nib.Nifti1Image(image_xyz, affine)
    nii.header.set_zooms(spacing_xyz)
    nii.header.set_xyzt_units(xyz="mm")
    nii.header["descrip"] = str(description).encode("ascii", errors="ignore")[:80]
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nii, str(path))


def _load_prepare_cfg(config_path: str) -> PrepareNNUNetCfg:
    module = load_python_module(Path(config_path), module_name="prepare_nnunet_cfg")
    raw = extract_python_cfg(module, Path(config_path))
    filtered = {k: v for k, v in raw.items() if k in {f.name for f in PrepareNNUNetCfg.__dataclass_fields__.values()}}
    return dataclass_from_raw(PrepareNNUNetCfg, filtered)


def _dataset_dir(cfg: PrepareNNUNetCfg) -> Path:
    return REPO_ROOT / cfg.nnunet_raw_dir / dataset_folder_name(cfg.dataset_id, cfg.dataset_name)


def _case_days_from_splits(cfg: PrepareNNUNetCfg) -> tuple[list[str], list[str], list[str]]:
    train_case_days = load_case_day_split(cfg.train_ids)
    val_case_days = load_case_day_split(cfg.val_ids)
    if not train_case_days:
        raise ValueError(f"Empty train split: {cfg.train_ids}")
    if not val_case_days:
        raise ValueError(f"Empty val split: {cfg.val_ids}")
    overlap = sorted(set(train_case_days) & set(val_case_days))
    if overlap:
        raise ValueError(f"train_ids and val_ids overlap: {overlap[:5]}")
    all_case_days = sorted(set(train_case_days) | set(val_case_days))
    return train_case_days, val_case_days, all_case_days


def _build_case_day_volumes(
    *,
    case_day: str,
    slice_paths: list[str],
    rle_index: dict[tuple[str, int], dict[str, str]],
    label_priority: list[str],
    spacing_z_override: float | None,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float], dict[str, Any]]:
    if not slice_paths:
        raise ValueError(f"slice_paths must not be empty for {case_day}")

    _, height, width, spacing_x, spacing_y = _parse_scan_filename(Path(slice_paths[0]).name)
    spacing = (float(spacing_x), float(spacing_y), float(spacing_x if spacing_z_override is None else spacing_z_override))
    image_volume = np.zeros((len(slice_paths), height, width), dtype=np.uint16)
    label_volume = np.zeros((len(slice_paths), height, width), dtype=np.uint8)

    overlap_slices = 0
    overwritten_voxels = 0
    source_class_voxels = {cls_name: 0 for cls_name in CLASSES}

    label_value_by_class = {cls_name: idx + 1 for idx, cls_name in enumerate(CLASSES)}

    for depth, slice_path in enumerate(slice_paths):
        _, expected_h, expected_w, current_spacing_x, current_spacing_y = _parse_scan_filename(Path(slice_path).name)
        if expected_h != height or expected_w != width:
            raise ValueError(
                f"Inconsistent dimensions in {slice_path}: expected {(height, width)}, got {(expected_h, expected_w)}"
            )
        if current_spacing_x != spacing_x or current_spacing_y != spacing_y:
            raise ValueError(
                f"Inconsistent spacing in {slice_path}: expected {(spacing_x, spacing_y)}, got {(current_spacing_x, current_spacing_y)}"
            )

        image = _read_scan(slice_path)
        if image.shape != (height, width):
            raise ValueError(f"Image shape mismatch in {slice_path}: expected {(height, width)}, got {image.shape}")
        image_volume[depth] = image.astype(np.uint16, copy=False)

        slice_idx = _parse_scan_filename(Path(slice_path).name)[0]
        per_class = rle_index.get((case_day, slice_idx), {})
        occupied = np.zeros((height, width), dtype=bool)
        had_overlap = False

        for cls_name in label_priority:
            mask = rle_decode(per_class.get(cls_name, ""), height, width).astype(bool)
            source_class_voxels[cls_name] += int(mask.sum())
            overlap_mask = occupied & mask
            overlap_count = int(overlap_mask.sum())
            if overlap_count > 0:
                had_overlap = True
                overwritten_voxels += overlap_count
            label_volume[depth][mask] = label_value_by_class[cls_name]
            occupied |= mask

        if had_overlap:
            overlap_slices += 1

    final_class_voxels = {
        cls_name: int((label_volume == label_value_by_class[cls_name]).sum()) for cls_name in CLASSES
    }
    stats = {
        "case_day": case_day,
        "num_slices": int(len(slice_paths)),
        "shape_zyx": [int(len(slice_paths)), int(height), int(width)],
        "spacing_xyz": [float(spacing[0]), float(spacing[1]), float(spacing[2])],
        "overlap_slices": int(overlap_slices),
        "overwritten_voxels": int(overwritten_voxels),
        "source_class_voxels": source_class_voxels,
        "final_class_voxels": final_class_voxels,
    }
    return image_volume, label_volume, spacing, stats


def prepare_nnunet_dataset(cfg: Any, *, logger=None) -> dict[str, Any]:
    raw_cfg = PrepareNNUNetCfg(
        data_root=str(getattr(cfg, "data_root")),
        train_csv=str(getattr(cfg, "train_csv")),
        train_ids=str(getattr(cfg, "train_ids")),
        val_ids=str(getattr(cfg, "val_ids")),
        dataset_id=int(getattr(cfg, "dataset_id")),
        dataset_name=str(getattr(cfg, "dataset_name")),
        nnunet_raw_dir=str(getattr(cfg, "nnunet_raw_dir")),
        label_collapse_mode=str(getattr(cfg, "label_collapse_mode", "priority_overwrite")),
        label_priority=[str(x) for x in list(getattr(cfg, "label_priority", list(CLASSES)))],
        spacing_z=getattr(cfg, "spacing_z", None),
        overwrite=bool(getattr(cfg, "overwrite", False)),
    )
    if raw_cfg.label_collapse_mode != "priority_overwrite":
        raise ValueError(f"Unsupported label_collapse_mode: {raw_cfg.label_collapse_mode}")
    if raw_cfg.label_priority != list(CLASSES):
        raise ValueError(
            f"label_priority must match {list(CLASSES)} to preserve the agreed baseline ordering, got {raw_cfg.label_priority}"
        )

    dataset_dir = _dataset_dir(raw_cfg)
    images_tr_dir = dataset_dir / "imagesTr"
    labels_tr_dir = dataset_dir / "labelsTr"
    images_tr_dir.mkdir(parents=True, exist_ok=True)
    labels_tr_dir.mkdir(parents=True, exist_ok=True)

    train_case_days, val_case_days, all_case_days = _case_days_from_splits(raw_cfg)
    train_dir = REPO_ROOT / raw_cfg.data_root / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    train_csv = REPO_ROOT / raw_cfg.train_csv
    if not train_csv.exists():
        raise FileNotFoundError(f"train_csv not found: {train_csv}")

    case_day_slices = build_case_day_slices(str(train_dir))
    rle_index = build_rle_index(str(train_csv))

    manifest_rows: list[dict[str, Any]] = []
    case_stats: list[dict[str, Any]] = []
    total_overlap_slices = 0
    total_overwritten_voxels = 0
    total_slices = 0
    source_class_voxels = {cls_name: 0 for cls_name in CLASSES}
    final_class_voxels = {cls_name: 0 for cls_name in CLASSES}

    for index, case_day in enumerate(all_case_days, start=1):
        if case_day not in case_day_slices:
            raise KeyError(f"case_day not found under {train_dir}: {case_day}")
        image_volume, label_volume, spacing, stats = _build_case_day_volumes(
            case_day=case_day,
            slice_paths=case_day_slices[case_day],
            rle_index=rle_index,
            label_priority=raw_cfg.label_priority,
            spacing_z_override=raw_cfg.spacing_z,
        )

        image_path = images_tr_dir / f"{case_day}_0000.nii.gz"
        label_path = labels_tr_dir / f"{case_day}.nii.gz"
        if raw_cfg.overwrite or not image_path.exists():
            _save_nifti(
                image_path,
                image_volume,
                spacing_xyz=spacing,
                description=f"UWGI {case_day} image",
            )
        if raw_cfg.overwrite or not label_path.exists():
            _save_nifti(
                label_path,
                label_volume,
                spacing_xyz=spacing,
                description=f"UWGI {case_day} multiclass label",
            )

        manifest_rows.append(
            {
                "case_day": case_day,
                "image_path": str(image_path),
                "label_path": str(label_path),
                "num_slices": int(stats["num_slices"]),
                "shape_d": int(stats["shape_zyx"][0]),
                "shape_h": int(stats["shape_zyx"][1]),
                "shape_w": int(stats["shape_zyx"][2]),
                "spacing_x": float(stats["spacing_xyz"][0]),
                "spacing_y": float(stats["spacing_xyz"][1]),
                "spacing_z": float(stats["spacing_xyz"][2]),
                "overlap_slices": int(stats["overlap_slices"]),
                "overwritten_voxels": int(stats["overwritten_voxels"]),
            }
        )
        case_stats.append(stats)
        total_slices += int(stats["num_slices"])
        total_overlap_slices += int(stats["overlap_slices"])
        total_overwritten_voxels += int(stats["overwritten_voxels"])
        for cls_name in CLASSES:
            source_class_voxels[cls_name] += int(stats["source_class_voxels"][cls_name])
            final_class_voxels[cls_name] += int(stats["final_class_voxels"][cls_name])

        if logger is not None:
            logger.info(
                "[%d/%d] Prepared %s -> image=%s label=%s overlap_slices=%d overwritten_voxels=%d",
                index,
                len(all_case_days),
                case_day,
                image_path.name,
                label_path.name,
                int(stats["overlap_slices"]),
                int(stats["overwritten_voxels"]),
            )

    dataset_json = {
        "channel_names": {"0": "grayscale"},
        "labels": {
            "background": 0,
            "large_bowel": 1,
            "small_bowel": 2,
            "stomach": 3,
        },
        "numTraining": int(len(all_case_days)),
        "file_ending": ".nii.gz",
    }
    dataset_json_path = dataset_dir / "dataset.json"
    dataset_json_path.write_text(json.dumps(dataset_json, indent=2) + "\n", encoding="utf-8")

    manifest_path = dataset_dir / "manifest.csv"
    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = list(manifest_rows[0].keys()) if manifest_rows else ["case_day", "image_path", "label_path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    collapse_summary = {
        "dataset_id": int(raw_cfg.dataset_id),
        "dataset_name": str(raw_cfg.dataset_name),
        "dataset_dir": str(dataset_dir),
        "num_case_days": int(len(all_case_days)),
        "num_train_case_days": int(len(train_case_days)),
        "num_val_case_days": int(len(val_case_days)),
        "total_slices": int(total_slices),
        "overlap_slices": int(total_overlap_slices),
        "overwritten_voxels": int(total_overwritten_voxels),
        "label_collapse_mode": str(raw_cfg.label_collapse_mode),
        "label_priority": list(raw_cfg.label_priority),
        "source_class_voxels": source_class_voxels,
        "final_class_voxels": final_class_voxels,
        "cases": case_stats,
    }
    collapse_summary_path = dataset_dir / "label_collapse_summary.json"
    collapse_summary_path.write_text(json.dumps(json_safe(collapse_summary), indent=2) + "\n", encoding="utf-8")

    config_snapshot_path = dataset_dir / "prepare_config.resolved.json"
    config_snapshot_path.write_text(
        json.dumps(cfg_to_serializable_dict(raw_cfg), indent=2) + "\n",
        encoding="utf-8",
    )

    return {
        "dataset_dir": dataset_dir,
        "dataset_json_path": dataset_json_path,
        "manifest_path": manifest_path,
        "collapse_summary_path": collapse_summary_path,
        "config_snapshot_path": config_snapshot_path,
        "train_case_days": train_case_days,
        "val_case_days": val_case_days,
        "all_case_days": all_case_days,
        "collapse_summary": collapse_summary,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare nnUNet v2 raw dataset for UWGI.")
    parser.add_argument("--config", type=str, default="configs/train_nnunet_baseline.py")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing dataset files.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    cfg = _load_prepare_cfg(args.config)
    if args.overwrite:
        cfg = PrepareNNUNetCfg(**{**cfg.__dict__, "overwrite": True})

    dataset_dir = _dataset_dir(cfg)
    logger = setup_logger(dataset_dir / "prepare_nnunet_dataset.log", name="prepare_nnunet_dataset")
    logger.info("Preparing nnUNet dataset under %s", dataset_dir)
    result = prepare_nnunet_dataset(cfg, logger=logger)
    logger.info("Wrote dataset.json          : %s", result["dataset_json_path"])
    logger.info("Wrote manifest              : %s", result["manifest_path"])
    logger.info("Wrote label collapse summary: %s", result["collapse_summary_path"])


if __name__ == "__main__":
    main()
