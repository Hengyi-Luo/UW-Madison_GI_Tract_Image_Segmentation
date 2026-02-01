from __future__ import annotations

import csv
import json
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


_scan_re = re.compile(r"slice_(\d+)_([0-9]+)_([0-9]+)_([0-9.]+)_([0-9.]+)\.png")


def parse_scan_filename(fname: str) -> Tuple[int, int, int]:
    """From slice_XXXX_H_W_psx_psy.png get slice index and nominal H,W in filename."""
    m = _scan_re.search(os.path.basename(fname))
    slice_idx = int(m.group(1))
    h = int(m.group(2))
    w = int(m.group(3))
    return slice_idx, h, w


def build_case_day_slices(root_dir: str) -> Dict[str, List[str]]:
    """Returns dict: key = 'case123_day0', value = sorted list of scan file paths."""
    out: Dict[str, List[str]] = {}
    for case in sorted(os.listdir(root_dir)):
        case_path = os.path.join(root_dir, case)

        for day in sorted(os.listdir(case_path)):
            day_path = os.path.join(case_path, day)
            scans_dir = os.path.join(day_path, "scans")

            files = [os.path.join(scans_dir, f) for f in os.listdir(scans_dir) if f.endswith(".png")]
            files.sort(key=lambda p: parse_scan_filename(p)[0])
            out[day] = files

    return out


def build_rle_index(train_csv_path: str) -> Dict[Tuple[str, int], Dict[str, str]]:
    """Map train.csv to idx[(case_day, slice_idx)] = {class_name: rle_string}."""
    df = pd.read_csv(train_csv_path)
    idx: Dict[Tuple[str, int], Dict[str, str]] = {}

    for _, row in df.iterrows():
        _id = row["id"]
        cls = row["class"]
        seg = row["segmentation"]

        parts = _id.split("_")
        case_day = "_".join(parts[0:2])
        slice_idx = int(parts[-1])
        key = (case_day, slice_idx)
        idx.setdefault(key, {})[cls] = seg

    return idx


def _normalize_case_day(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    # Kaggle train.csv id format: case123_day0_slice_0001
    if "_slice_" in s:
        parts = s.split("_")
        if len(parts) >= 2:
            return "_".join(parts[0:2])
    return s


def _iter_ids_from_csv(path: str, column: Optional[str]) -> Iterable[str]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []
        col = column
        if not col:
            for cand in ("case_day", "case_day_id", "id"):
                if cand in reader.fieldnames:
                    col = cand
                    break
        if not col:
            col = reader.fieldnames[0]
        for row in reader:
            v = row.get(col, "")
            if v is not None:
                yield str(v)


def load_case_days(ids_path: str, column: Optional[str] = None) -> List[str]:
    """
    Load case_day identifiers from a file.

    Supported formats:
    - .csv: detects a `case_day`/`id`-like column, or use `column=...`
    - .json: a list of strings, or a dict with one of `val`/`case_days`/`ids`
    - other: treated as a text file with one ID per line

    Each entry may be either:
    - `case123_day0`
    - Kaggle `case123_day0_slice_0001` (normalized to `case123_day0`)
    """
    if not ids_path:
        raise ValueError("ids_path is required")
    if not os.path.exists(ids_path):
        raise FileNotFoundError(f"IDs file not found: {ids_path}")

    raw: List[str] = []
    if ids_path.endswith(".json"):
        with open(ids_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            raw = [str(x) for x in obj]
        elif isinstance(obj, dict):
            for k in ("val", "case_days", "ids", "train"):
                if k in obj and isinstance(obj[k], list):
                    raw = [str(x) for x in obj[k]]
                    break
            else:
                raise ValueError(f"Unrecognized json shape in {ids_path}: expected list or dict with val/case_days/ids")
        else:
            raise ValueError(f"Unrecognized json shape in {ids_path}: {type(obj)}")
    elif ids_path.endswith(".csv"):
        raw = list(_iter_ids_from_csv(ids_path, column=column))
    else:
        with open(ids_path, "r", encoding="utf-8") as f:
            raw = [ln.strip() for ln in f.readlines()]

    out: List[str] = []
    seen = set()
    for s in raw:
        s = _normalize_case_day(str(s))
        if not s or s.startswith("#"):
            continue
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out
