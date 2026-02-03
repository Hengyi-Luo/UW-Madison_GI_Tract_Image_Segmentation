#!/usr/bin/env python3
"""
Create train/val split files and val visualization samples.

Defaults:
- data_root: /home/jeremiah/github/UW-Madison_GI_Tract_Image_Segmentation/inputs
- split ratio: 0.2
- outputs to: <data_root>/splits/

Outputs:
- train_case_days.csv
- val_case_days.csv
- val_vis_samples.json (same format as scripts/select_eval_samples.py)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
from typing import Dict, List, Optional, Set, Tuple

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.uwgi.data_utils import build_case_day_slices
from src.uwgi.splits import split_case_days


CLASSES = ("large_bowel", "small_bowel", "stomach")

_slice_re = re.compile(r"slice_(\d+)_([0-9]+)_([0-9]+)_([0-9.]+)_([0-9.]+)\.png$")


def _parse_slice_idx(path: str) -> int:
    m = _slice_re.search(os.path.basename(path))
    if not m:
        raise ValueError(f"Unrecognized slice filename: {path}")
    return int(m.group(1))


def _read_case_days_from_train_csv(train_csv: str) -> List[str]:
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"train.csv not found: {train_csv}")
    out: List[str] = []
    seen: Set[str] = set()
    with open(train_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            _id = str(row.get("id", "") or "")
            if not _id:
                continue
            parts = _id.split("_")
            if len(parts) < 2:
                continue
            case_day = "_".join(parts[0:2])
            if case_day not in seen:
                seen.add(case_day)
                out.append(case_day)
    return out


def _write_case_days_csv(path: str, case_days: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_day"])
        for cd in case_days:
            w.writerow([cd])


def _rle_area(rle: str) -> int:
    if rle is None:
        return 0
    s = str(rle).strip()
    if not s or s == "nan":
        return 0
    parts = s.split()
    if len(parts) < 2:
        return 0
    total = 0
    for ln in parts[1::2]:
        try:
            total += int(float(ln))
        except Exception:
            return 0
    return int(total)


def _load_slice_class_areas(
    train_csv: str,
    allowed_case_days: set[str],
) -> Dict[Tuple[str, int], Dict[str, int]]:
    areas: Dict[Tuple[str, int], Dict[str, int]] = {}
    with open(train_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            _id = str(row.get("id", "") or "")
            cls = str(row.get("class", "") or "")
            seg = row.get("segmentation", "")
            if not _id or not cls:
                continue
            parts = _id.split("_")
            if len(parts) < 3:
                continue
            case_day = "_".join(parts[0:2])
            if case_day not in allowed_case_days:
                continue
            try:
                slice_idx = int(parts[-1])
            except Exception:
                continue
            key = (case_day, slice_idx)
            areas.setdefault(key, {})[cls] = _rle_area(seg)
    return areas


def _pick_val_vis_samples(
    *,
    val_days: List[str],
    case_day_slices: Dict[str, List[str]],
    areas: Dict[Tuple[str, int], Dict[str, int]],
    n: int,
    min_area: int,
    seed: int,
) -> List[dict]:
    rng = random.Random(int(seed))
    candidates: List[dict] = []
    for cd in val_days:
        slice_paths = case_day_slices.get(cd) or []
        if not slice_paths:
            continue
        for slice_pos, p in enumerate(slice_paths):
            slice_idx = int(_parse_slice_idx(p))
            per_cls = areas.get((cd, int(slice_idx)), {})
            per_cls = {c: int(per_cls.get(c, 0)) for c in CLASSES}
            present = sum(per_cls[c] >= int(min_area) for c in CLASSES)
            if present < 3:
                continue
            total_area = int(sum(per_cls.values()))
            candidates.append(
                {
                    "case": str(cd).split("_", 1)[0],
                    "case_day": str(cd),
                    "slice_idx": int(slice_idx),
                    "slice_pos": int(slice_pos),
                    "slice_path": str(p),
                    "present_classes": int(present),
                    "total_area": int(total_area),
                    "areas": per_cls,
                    "min_area": int(min_area),
                    "min_classes": 3,
                }
            )

    rng.shuffle(candidates)
    return candidates[: int(n)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/home/jeremiah/github/UW-Madison_GI_Tract_Image_Segmentation/inputs")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-vis-n", type=int, default=16)
    p.add_argument("--val-vis-min-area", type=int, default=1)
    p.add_argument(
        "--allow-less",
        action="store_true",
        help="Allow fewer than N samples if dataset has too few 3-label slices.",
    )
    args = p.parse_args()

    data_root = str(args.data_root)
    train_csv = os.path.join(data_root, "train.csv")
    train_dir = os.path.join(data_root, "train")
    split_dir = os.path.join(data_root, "splits")

    case_days = _read_case_days_from_train_csv(train_csv)
    if not case_days:
        raise RuntimeError("No case_days found in train.csv")

    train_days, val_days = split_case_days(case_days, val_ratio=args.val_ratio, seed=args.seed)

    train_csv_out = os.path.join(split_dir, "train_case_days.csv")
    val_csv_out = os.path.join(split_dir, "val_case_days.csv")
    _write_case_days_csv(train_csv_out, train_days)
    _write_case_days_csv(val_csv_out, val_days)

    case_day_slices = build_case_day_slices(train_dir)
    areas = _load_slice_class_areas(train_csv, allowed_case_days=set(val_days))
    val_vis_samples = _pick_val_vis_samples(
        val_days=val_days,
        case_day_slices=case_day_slices,
        areas=areas,
        n=int(args.val_vis_n),
        min_area=int(args.val_vis_min_area),
        seed=int(args.seed),
    )
    val_vis_path = os.path.join(split_dir, "val_vis_samples.json")
    with open(val_vis_path, "w", encoding="utf-8") as f:
        payload = {
            "criteria": {"min_classes": 3, "min_area": int(args.val_vis_min_area)},
            "n_requested": int(args.val_vis_n),
            "samples": val_vis_samples,
        }
        json.dump(payload, f, indent=2, ensure_ascii=False)

    if len(val_vis_samples) < int(args.val_vis_n):
        msg = (
            f"Requested n={int(args.val_vis_n)} but only found {len(val_vis_samples)} slices "
            f"with 3 labels (min_area={int(args.val_vis_min_area)})."
        )
        if not bool(args.allow_less):
            raise RuntimeError(msg)
        print(f"[WARN] {msg}")

    print(f"Train case_days: {len(train_days)}")
    print(f"Val case_days: {len(val_days)}")
    print(f"Wrote: {train_csv_out}")
    print(f"Wrote: {val_csv_out}")
    print(f"Wrote: {val_vis_path} ({len(val_vis_samples)} samples)")


if __name__ == "__main__":
    main()
