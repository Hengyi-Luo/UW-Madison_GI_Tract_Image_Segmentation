#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from src.constants import CLASS2IDX, CLASSES  # noqa: E402
from src.data_utils import build_case_day_slices  # noqa: E402
from src.rle import rle_decode  # noqa: E402


def _parse_scan_filename(filename: str) -> tuple[int, int, int, float, float]:
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 6:
        raise ValueError(f"Unexpected scan filename: {filename}")
    return int(parts[1]), int(parts[3]), int(parts[2]), float(parts[4]), float(parts[5])


def _read_scan(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to read scan: {path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _load_case_rles(csv_path: Path, case_day: str) -> dict[tuple[str, int], dict[str, str]]:
    index: dict[tuple[str, int], dict[str, str]] = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"id", "class", "segmentation"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"CSV must contain columns {sorted(required)}: {csv_path}")
        for row in reader:
            sample_id = str(row.get("id", "")).strip()
            cls_name = str(row.get("class", "")).strip()
            if cls_name not in CLASS2IDX or not sample_id:
                continue
            parts = sample_id.split("_")
            if len(parts) < 4:
                continue
            row_case_day = "_".join(parts[:2])
            if row_case_day != case_day:
                continue
            slice_idx = int(parts[-1])
            index.setdefault((row_case_day, slice_idx), {})[cls_name] = row.get("segmentation", "")
    return index


def _build_volumes(
    case_day: str,
    data_root: Path,
    masks_csv: Path | None,
) -> tuple[np.ndarray, dict[str, np.ndarray], tuple[float, float, float]]:
    train_dir = data_root / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    case_day_slices = build_case_day_slices(str(train_dir))
    if case_day not in case_day_slices:
        raise KeyError(f"case_day not found under {train_dir}: {case_day}")

    slice_paths = case_day_slices[case_day]
    _, height, width, spacing_x, spacing_y = _parse_scan_filename(Path(slice_paths[0]).name)

    image_volume = np.zeros((len(slice_paths), height, width), dtype=np.uint16)
    label_volumes = {
        cls_name: np.zeros((len(slice_paths), height, width), dtype=np.uint8) for cls_name in CLASSES
    }
    rle_index = _load_case_rles(masks_csv, case_day) if masks_csv is not None else {}

    for depth, slice_path in enumerate(slice_paths):
        image = _read_scan(slice_path)
        if image.shape != (height, width):
            raise ValueError(f"Inconsistent image shape in {slice_path}: expected {(height, width)}, got {image.shape}")
        image_volume[depth] = image

        slice_idx, _, _, _, _ = _parse_scan_filename(Path(slice_path).name)
        per_class = rle_index.get((case_day, slice_idx), {})
        for cls_name in CLASSES:
            decoded = rle_decode(per_class.get(cls_name, ""), height, width)
            label_volumes[cls_name][depth] = decoded.astype(np.uint8)

    return image_volume, label_volumes, (float(spacing_x), float(spacing_y), float(spacing_x))


def _save_nifti(path: Path, volume_zyx: np.ndarray, spacing_xyz: tuple[float, float, float]) -> None:
    affine = np.diag([spacing_xyz[0], spacing_xyz[1], spacing_xyz[2], 1.0]).astype(np.float32)
    # NIfTI is stored as (X, Y, Z); project data is built as (Z, Y, X).
    image = nib.Nifti1Image(np.transpose(volume_zyx, (2, 1, 0)), affine)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(path))


def _write_legend(path: Path) -> None:
    rows = ["class_name,file_suffix"]
    rows.extend(f"{name},mask_{name}.nii.gz" for name in CLASSES)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export one case_day as NIfTI image and label volumes for 3D Slicer.")
    parser.add_argument("--case-day", required=True, help="case_day like case122_day27")
    parser.add_argument("--data-root", default="inputs", help="Dataset root containing train/")
    parser.add_argument(
        "--masks-csv",
        default="inputs/train.csv",
        help="Kaggle-format CSV with columns id,class,segmentation. Use '' to export only the raw image volume.",
    )
    parser.add_argument("--out-dir", default="outputs/nifti_export", help="Output directory")
    parser.add_argument("--prefix", default="", help="Optional filename prefix; defaults to case_day")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    case_day = str(args.case_day).strip()
    masks_csv = Path(args.masks_csv) if str(args.masks_csv).strip() else None
    if masks_csv is not None and not masks_csv.exists():
        raise FileNotFoundError(f"masks_csv not found: {masks_csv}")

    image_volume, label_volumes, spacing = _build_volumes(case_day, Path(args.data_root), masks_csv)

    out_dir = Path(args.out_dir)
    prefix = str(args.prefix).strip() or case_day
    image_path = out_dir / f"{prefix}_image.nii.gz"
    legend_path = out_dir / f"{prefix}_labels_legend.csv"

    _save_nifti(image_path, image_volume, spacing)
    label_paths: list[Path] = []
    for cls_name in CLASSES:
        label_path = out_dir / f"{prefix}_mask_{cls_name}.nii.gz"
        _save_nifti(label_path, label_volumes[cls_name], spacing)
        label_paths.append(label_path)
    _write_legend(legend_path)

    print(f"Wrote image: {image_path}")
    for label_path in label_paths:
        print(f"Wrote mask: {label_path}")
    print(f"Wrote legend: {legend_path}")
    print(f"Spacing used (x, y, z): {spacing}")


if __name__ == "__main__":
    main()
