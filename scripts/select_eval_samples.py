#!/usr/bin/env python3
"""
Select representative validation slices for qualitative evaluation.

Current requirement:
  - MUST select exactly N samples
  - each selected 2D slice MUST contain at least 3 labels (all 3 classes present)
  - area is not important (use --min_area to tweak, default is minimal)

Outputs are written under:
  ./input/uw-madison-gi-tract-image-segmentation/splits
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from typing import Dict, Iterable, List, Optional, Tuple


CLASSES = ("large_bowel", "small_bowel", "stomach")

_slice_re = re.compile(r"slice_(\d+)_([0-9]+)_([0-9]+)_([0-9.]+)_([0-9.]+)\.png$")


def _parse_slice_idx(path: str) -> int:
    m = _slice_re.search(os.path.basename(path))
    if not m:
        raise ValueError(f"Unrecognized slice filename: {path}")
    return int(m.group(1))


def _case_id(case_day: str) -> str:
    # case123_day0 -> case123
    s = str(case_day).strip()
    return s.split("_", 1)[0]


def _read_case_days(path: str, column: Optional[str] = None) -> List[str]:
    if not path:
        raise ValueError("--val_ids is required")
    if not os.path.exists(path):
        raise FileNotFoundError(f"val_ids not found: {path}")

    def normalize(v: str) -> str:
        v = (v or "").strip()
        if not v or v.startswith("#"):
            return ""
        # Kaggle style: case123_day0_slice_0001 -> case123_day0
        if "_slice_" in v:
            parts = v.split("_")
            if len(parts) >= 2:
                return "_".join(parts[0:2])
        return v

    out: List[str] = []
    seen = set()
    if path.endswith(".json"):
        obj = json.load(open(path, "r", encoding="utf-8"))
        if isinstance(obj, list):
            raw = [str(x) for x in obj]
        elif isinstance(obj, dict):
            for k in ("val", "case_days", "ids", "train"):
                if k in obj and isinstance(obj[k], list):
                    raw = [str(x) for x in obj[k]]
                    break
            else:
                raise ValueError(f"Unrecognized json shape in {path}")
        else:
            raise ValueError(f"Unrecognized json shape in {path}")
        for s in raw:
            s = normalize(s)
            if s and s not in seen:
                out.append(s)
                seen.add(s)
        return out

    if path.endswith(".csv"):
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return []
            col = (column or "").strip() or None
            if not col:
                for cand in ("case_day", "case_day_id", "id"):
                    if cand in reader.fieldnames:
                        col = cand
                        break
            if not col:
                col = reader.fieldnames[0]
            for row in reader:
                s = normalize(str(row.get(col, "") or ""))
                if s and s not in seen:
                    out.append(s)
                    seen.add(s)
        return out

    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            s = normalize(ln)
            if s and s not in seen:
                out.append(s)
                seen.add(s)
    return out


def _build_case_day_slices(train_dir: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for case in sorted(os.listdir(train_dir)):
        case_path = os.path.join(train_dir, case)
        if not os.path.isdir(case_path):
            continue
        for day in sorted(os.listdir(case_path)):
            day_path = os.path.join(case_path, day)
            scans_dir = os.path.join(day_path, "scans")
            if not os.path.isdir(scans_dir):
                continue
            files = [os.path.join(scans_dir, f) for f in os.listdir(scans_dir) if f.endswith(".png")]
            files.sort(key=_parse_slice_idx)
            out[day] = files
    return out


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
    # starts are parts[0::2], lengths are parts[1::2]
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
    """
    Returns areas[(case_day, slice_idx)] = {class_name: area_int}
    Only keeps rows whose case_day is in allowed_case_days.
    """
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


def _pick_samples(
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
                    "case": _case_id(cd),
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

    # Deterministic shuffle then take first N to ensure we can always "fill" if enough exist.
    rng.shuffle(candidates)
    return candidates[: int(n)]


def _assert_inside_input_splits(out_path: str) -> None:
    out_abs = os.path.abspath(out_path)
    allowed = os.path.abspath(
        os.path.join(".", "input", "uw-madison-gi-tract-image-segmentation", "splits")
    )
    if not (out_abs == allowed or out_abs.startswith(allowed + os.sep)):
        raise ValueError(f"This script only writes under {allowed}. Got: {out_abs}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./input/uw-madison-gi-tract-image-segmentation")
    p.add_argument("--val_ids", type=str, required=True)
    p.add_argument("--ids_column", type=str, default="")
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--min_area", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out",
        type=str,
        default="./input/uw-madison-gi-tract-image-segmentation/splits/val_vis_samples.json",
    )
    p.add_argument(
        "--csv_out",
        type=str,
        default="./input/uw-madison-gi-tract-image-segmentation/splits/val_vis_samples.csv",
    )
    p.add_argument(
        "--allow_less",
        action="store_true",
        help="Allow fewer than N samples if dataset has too few 3-label slices.",
    )
    args = p.parse_args()

    _assert_inside_input_splits(str(args.out))
    if args.csv_out:
        _assert_inside_input_splits(str(args.csv_out))

    data_root = str(args.data_root)
    train_dir = os.path.join(data_root, "train")
    train_csv = os.path.join(data_root, "train.csv")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"train dir not found: {train_dir}")
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"train.csv not found: {train_csv}")

    val_days = _read_case_days(str(args.val_ids), column=(args.ids_column or None))
    case_day_slices = _build_case_day_slices(train_dir)
    val_days = [cd for cd in val_days if cd in case_day_slices]
    if not val_days:
        raise ValueError("No valid val case_days after filtering by train/ directory.")

    areas = _load_slice_class_areas(train_csv, allowed_case_days=set(val_days))

    # Must be >=3 labels per 2D slice. Area is minimal by default.
    samples = _pick_samples(
        val_days=val_days,
        case_day_slices=case_day_slices,
        areas=areas,
        n=int(args.n),
        min_area=int(args.min_area),
        seed=int(args.seed),
    )
    used = {"min_classes": 3, "min_area": int(args.min_area)}

    print(f"val case_days: {len(val_days)}")
    print(f"selected samples: {len(samples)} used={used}")
    if len(samples) < int(args.n):
        msg = (
            f"Requested n={int(args.n)} but only found {len(samples)} slices with 3 labels "
            f"(min_area={int(args.min_area)})."
        )
        if not bool(args.allow_less):
            raise RuntimeError(msg)
        print(f"[WARN] {msg}")

    os.makedirs(os.path.dirname(str(args.out)) or ".", exist_ok=True)
    payload = {"criteria": used, "n_requested": int(args.n), "samples": samples}
    with open(str(args.out), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"saved: {args.out}")

    if args.csv_out:
        os.makedirs(os.path.dirname(str(args.csv_out)) or ".", exist_ok=True)
        with open(str(args.csv_out), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "case",
                    "case_day",
                    "slice_idx",
                    "slice_pos",
                    "slice_path",
                    "present_classes",
                    "total_area",
                    "min_area",
                    "min_classes",
                ],
            )
            w.writeheader()
            for s in samples:
                w.writerow({k: s.get(k, "") for k in w.fieldnames})
        print(f"saved: {args.csv_out}")


if __name__ == "__main__":
    main()
