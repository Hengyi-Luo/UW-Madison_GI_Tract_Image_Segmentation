#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from dataclasses import dataclass

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.constants import CLASSES
from src.data_utils import build_case_day_slices, load_case_days, parse_scan_filename


def _rle_area(seg: str | None) -> int:
    if seg is None:
        return 0
    seg = str(seg).strip()
    if not seg or seg.lower() == "nan":
        return 0
    parts = seg.split()
    if len(parts) % 2 != 0:
        return 0
    try:
        return int(sum(int(x) for x in parts[1::2]))
    except Exception:
        return 0


@dataclass(frozen=True)
class Sample:
    case_day: str
    slice_idx: int
    areas: dict[str, int]

    @property
    def case(self) -> str:
        return self.case_day.split("_", 1)[0]

    @property
    def present_classes(self) -> int:
        return sum(1 for a in self.areas.values() if int(a) > 0)

    @property
    def total_area(self) -> int:
        return int(sum(int(a) for a in self.areas.values()))


def _choose_quantile_samples(candidates: list[Sample], n: int, seed: int) -> list[Sample]:
    if n <= 0:
        raise ValueError(f"n must be > 0 (got {n})")
    if len(candidates) < n:
        raise ValueError(f"Not enough candidates ({len(candidates)}) to sample n={n}.")

    candidates = sorted(candidates, key=lambda s: (s.total_area, s.case_day, s.slice_idx), reverse=True)
    idxs = np.linspace(0, len(candidates) - 1, num=n).round().astype(int).tolist()

    # Ensure unique selections while preserving order.
    seen: set[int] = set()
    uniq: list[int] = []
    for i in idxs:
        if i in seen:
            continue
        seen.add(i)
        uniq.append(i)
    if len(uniq) < n:
        for i in range(len(candidates)):
            if i in seen:
                continue
            seen.add(i)
            uniq.append(i)
            if len(uniq) >= n:
                break

    selected = [candidates[i] for i in uniq[:n]]
    rng = random.Random(int(seed))
    rng.shuffle(selected)
    return selected


def _slice_pos_for(case_day_slices: dict[str, list[str]], case_day: str, slice_idx: int) -> int:
    slice_paths = case_day_slices[case_day]
    guess = int(slice_idx) - 1
    if 0 <= guess < len(slice_paths):
        _, parsed_idx, _, _ = parse_scan_filename(os.path.basename(slice_paths[guess]))
        if int(parsed_idx) == int(slice_idx):
            return guess

    # Fallback: scan for exact match
    for pos, p in enumerate(slice_paths):
        _, parsed_idx, _, _ = parse_scan_filename(os.path.basename(p))
        if int(parsed_idx) == int(slice_idx):
            return int(pos)
    raise KeyError(f"slice_idx={slice_idx} not found in {case_day} scan list")


def _kaggle_style_slice_path(kaggle_input_root: str, case: str, case_day: str, filename: str) -> str:
    kaggle_input_root = kaggle_input_root.rstrip("/")
    return f"{kaggle_input_root}/train/{case}/{case_day}/scans/{filename}"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate visualization sample lists from a split case_day CSV.")
    p.add_argument("--data_root", type=str, default="./inputs")
    p.add_argument("--train_csv", type=str, default="./inputs/train.csv")
    p.add_argument("--split_ids", type=str, default="./inputs/splits/train_case_days.csv")
    p.add_argument("--out_json", type=str, default="./inputs/splits/train_vis_samples.json")
    p.add_argument("--out_csv", type=str, default="./inputs/splits/train_vis_samples.csv")
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--min_classes", type=int, default=3)
    p.add_argument("--min_area", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--kaggle_input_root",
        type=str,
        default="./input/uw-madison-gi-tract-image-segmentation",
        help="Prefix for the slice_path field (does not need to exist locally).",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    split_case_days = set(load_case_days(args.split_ids))
    if not split_case_days:
        raise ValueError(f"No case_days found in split_ids: {args.split_ids}")

    train_dir = os.path.join(args.data_root, "train")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Train dir not found: {train_dir}")
    if not os.path.exists(args.train_csv):
        raise FileNotFoundError(f"train_csv not found: {args.train_csv}")

    case_day_slices = build_case_day_slices(train_dir)
    split_case_days = {cd for cd in split_case_days if cd in case_day_slices}
    if not split_case_days:
        raise ValueError("No split case_days exist in data_root/train.")

    # Aggregate areas per (case_day, slice_idx, class)
    areas_by_key: dict[tuple[str, int], dict[str, int]] = {}
    with open(args.train_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            _id = str(row.get("id", "")).strip()
            cls = str(row.get("class", "")).strip()
            seg = row.get("segmentation", "")
            if not _id or cls not in CLASSES:
                continue

            parts = _id.split("_")
            if len(parts) < 4:
                continue
            case_day = "_".join(parts[0:2])
            if case_day not in split_case_days:
                continue
            try:
                slice_idx = int(parts[-1])
            except Exception:
                continue

            key = (case_day, int(slice_idx))
            areas = areas_by_key.get(key)
            if areas is None:
                areas = {c: 0 for c in CLASSES}
                areas_by_key[key] = areas
            areas[cls] = int(_rle_area(seg))

    candidates: list[Sample] = []
    min_classes = int(args.min_classes)
    min_area = int(args.min_area)
    for (case_day, slice_idx), areas in areas_by_key.items():
        present = sum(1 for a in areas.values() if int(a) > 0)
        if present < min_classes:
            continue
        if min_area > 0 and any(int(areas[c]) < min_area for c in CLASSES):
            continue
        candidates.append(Sample(case_day=case_day, slice_idx=int(slice_idx), areas=dict(areas)))

    selected = _choose_quantile_samples(candidates, n=int(args.n), seed=int(args.seed))

    samples_out: list[dict] = []
    for s in selected:
        slice_pos = _slice_pos_for(case_day_slices, s.case_day, s.slice_idx)
        local_slice_path = case_day_slices[s.case_day][slice_pos]
        filename = os.path.basename(local_slice_path)

        samples_out.append(
            {
                "case": s.case,
                "case_day": s.case_day,
                "slice_idx": int(s.slice_idx),
                "slice_pos": int(slice_pos),
                "slice_path": _kaggle_style_slice_path(args.kaggle_input_root, s.case, s.case_day, filename),
                "present_classes": int(s.present_classes),
                "total_area": int(s.total_area),
                "areas": {c: int(s.areas.get(c, 0)) for c in CLASSES},
                "min_area": int(min_area),
                "min_classes": int(min_classes),
            }
        )

    out_json = {
        "criteria": {"min_classes": int(min_classes), "min_area": int(min_area)},
        "n_requested": int(args.n),
        "samples": samples_out,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)
        f.write("\n")

    fieldnames = [
        "case",
        "case_day",
        "slice_idx",
        "slice_pos",
        "slice_path",
        "present_classes",
        "total_area",
        "min_area",
        "min_classes",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in samples_out:
            w.writerow({k: r.get(k) for k in fieldnames})

    print(f"Wrote {len(samples_out)} samples:")
    print(f"  JSON: {args.out_json}")
    print(f"  CSV : {args.out_csv}")


if __name__ == "__main__":
    main()
