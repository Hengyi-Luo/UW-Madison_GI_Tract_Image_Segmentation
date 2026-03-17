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

from src.data_utils import build_case_day_slices  # noqa: E402


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


def _build_volume(slice_paths: list[str]) -> tuple[np.ndarray, tuple[float, float, float]]:
    if not slice_paths:
        raise ValueError("slice_paths must not be empty")

    _, height, width, spacing_x, spacing_y = _parse_scan_filename(Path(slice_paths[0]).name)
    volume = np.zeros((len(slice_paths), height, width), dtype=np.uint16)

    for depth, slice_path in enumerate(slice_paths):
        _, expected_h, expected_w, current_spacing_x, current_spacing_y = _parse_scan_filename(Path(slice_path).name)
        if expected_h != height or expected_w != width:
            raise ValueError(
                f"Inconsistent dimensions in {slice_path}: expected {(height, width)}, got {(expected_h, expected_w)}"
            )
        if current_spacing_x != spacing_x or current_spacing_y != spacing_y:
            raise ValueError(
                f"Inconsistent in-plane spacing in {slice_path}: expected {(spacing_x, spacing_y)}, "
                f"got {(current_spacing_x, current_spacing_y)}"
            )
        image = _read_scan(slice_path)
        if image.shape != (height, width):
            raise ValueError(f"Image shape mismatch in {slice_path}: expected {(height, width)}, got {image.shape}")
        volume[depth] = image

    return volume, (float(spacing_x), float(spacing_y), float(spacing_x))


def _save_nifti(
    path: Path,
    volume_zyx: np.ndarray,
    *,
    spacing_xyz: tuple[float, float, float],
    case_day: str,
) -> None:
    affine = np.diag([spacing_xyz[0], spacing_xyz[1], spacing_xyz[2], 1.0]).astype(np.float32)
    image_xyz = np.transpose(volume_zyx, (2, 1, 0))
    nii = nib.Nifti1Image(image_xyz, affine)
    nii.header.set_zooms(spacing_xyz)
    nii.header.set_xyzt_units(xyz="mm")
    nii.header["descrip"] = f"UWGI {case_day} from PNG stack".encode("ascii", errors="ignore")[:80]
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nii, str(path))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert all train PNG stacks into case_day NIfTI volumes.")
    parser.add_argument("--data-root", default="inputs", help="Dataset root containing train/")
    parser.add_argument("--out-dir", default="inputs/train_nifti", help="Directory for NIfTI outputs")
    parser.add_argument(
        "--case-day",
        default="",
        help="Optional single case_day to export. If empty, exports every case_day under data_root/train.",
    )
    parser.add_argument(
        "--spacing-z",
        type=float,
        default=None,
        help="Optional z spacing override. If omitted, z spacing follows the PNG in-plane spacing.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing NIfTI files")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    train_dir = Path(args.data_root) / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    case_day_slices = build_case_day_slices(str(train_dir))
    case_days = sorted(case_day_slices.keys())
    if args.case_day:
        case_day = str(args.case_day).strip()
        if case_day not in case_day_slices:
            raise KeyError(f"case_day not found under {train_dir}: {case_day}")
        case_days = [case_day]

    out_dir = Path(args.out_dir)
    manifest_path = out_dir / "manifest.csv"
    manifest_rows: list[dict[str, str | int | float]] = []

    for index, case_day in enumerate(case_days, start=1):
        slice_paths = case_day_slices[case_day]
        volume, spacing = _build_volume(slice_paths)
        if args.spacing_z is not None:
            spacing = (spacing[0], spacing[1], float(args.spacing_z))

        case = case_day.split("_", 1)[0]
        nifti_path = out_dir / case / f"{case_day}.nii.gz"
        if nifti_path.exists() and not bool(args.overwrite):
            print(f"[{index}/{len(case_days)}] Skip existing: {nifti_path}")
        else:
            _save_nifti(nifti_path, volume, spacing_xyz=spacing, case_day=case_day)
            print(f"[{index}/{len(case_days)}] Wrote {nifti_path}")

        manifest_rows.append(
            {
                "case": case,
                "case_day": case_day,
                "num_slices": int(volume.shape[0]),
                "height": int(volume.shape[1]),
                "width": int(volume.shape[2]),
                "spacing_x": float(spacing[0]),
                "spacing_y": float(spacing[1]),
                "spacing_z": float(spacing[2]),
                "nifti_path": str(nifti_path),
            }
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "case",
            "case_day",
            "num_slices",
            "height",
            "width",
            "spacing_x",
            "spacing_y",
            "spacing_z",
            "nifti_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
