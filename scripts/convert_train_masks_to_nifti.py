#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from src.constants import CLASSES  # noqa: E402
from src.data_utils import build_case_day_slices, build_rle_index  # noqa: E402
from src.rle import rle_decode  # noqa: E402


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


def _save_nifti(path: Path, volume_zyx: np.ndarray, spacing_xyz: tuple[float, float, float], *, description: str) -> None:
    affine = np.diag([spacing_xyz[0], spacing_xyz[1], spacing_xyz[2], 1.0]).astype(np.float32)
    image_xyz = np.transpose(volume_zyx, (2, 1, 0))
    nii = nib.Nifti1Image(image_xyz, affine)
    nii.header.set_zooms(spacing_xyz)
    nii.header.set_xyzt_units(xyz="mm")
    nii.header["descrip"] = str(description).encode("ascii", errors="ignore")[:80]
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nii, str(path))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert all train.csv masks into per-class NIfTI volumes.")
    parser.add_argument("--data-root", default="inputs", help="Dataset root containing train/ and train.csv")
    parser.add_argument("--train-csv", default="", help="Optional explicit path to train.csv")
    parser.add_argument("--out-dir", default="inputs/train_mask_nifti", help="Directory for mask NIfTI outputs")
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

    data_root = Path(args.data_root)
    train_dir = data_root / "train"
    train_csv = Path(args.train_csv) if str(args.train_csv).strip() else data_root / "train.csv"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    if not train_csv.exists():
        raise FileNotFoundError(f"train_csv not found: {train_csv}")

    case_day_slices = build_case_day_slices(str(train_dir))
    rle_index = build_rle_index(str(train_csv))

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
        if not slice_paths:
            continue
        _, height, width, spacing_x, spacing_y = _parse_scan_filename(Path(slice_paths[0]).name)
        spacing = (float(spacing_x), float(spacing_y), float(spacing_x))
        if args.spacing_z is not None:
            spacing = (spacing[0], spacing[1], float(args.spacing_z))

        label_volumes = {
            cls_name: np.zeros((len(slice_paths), height, width), dtype=np.uint8) for cls_name in CLASSES
        }
        for depth, slice_path in enumerate(slice_paths):
            slice_idx, _, _, _, _ = _parse_scan_filename(Path(slice_path).name)
            per_class = rle_index.get((case_day, slice_idx), {})
            for cls_name in CLASSES:
                label_volumes[cls_name][depth] = rle_decode(per_class.get(cls_name, ""), height, width).astype(np.uint8)

        case = case_day.split("_", 1)[0]
        case_dir = out_dir / case
        wrote_any = False
        for cls_name in CLASSES:
            mask_path = case_dir / f"{case_day}_mask_{cls_name}.nii.gz"
            if mask_path.exists() and not bool(args.overwrite):
                continue
            _save_nifti(
                mask_path,
                label_volumes[cls_name],
                spacing,
                description=f"UWGI {case_day} mask {cls_name}",
            )
            wrote_any = True

        if wrote_any:
            print(f"[{index}/{len(case_days)}] Wrote masks for {case_day}")
        else:
            print(f"[{index}/{len(case_days)}] Skip existing masks for {case_day}")

        for cls_name in CLASSES:
            manifest_rows.append(
                {
                    "case": case,
                    "case_day": case_day,
                    "class_name": cls_name,
                    "num_slices": int(len(slice_paths)),
                    "height": int(height),
                    "width": int(width),
                    "spacing_x": float(spacing[0]),
                    "spacing_y": float(spacing[1]),
                    "spacing_z": float(spacing[2]),
                    "nifti_path": str(case_dir / f"{case_day}_mask_{cls_name}.nii.gz"),
                }
            )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "case",
            "case_day",
            "class_name",
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
