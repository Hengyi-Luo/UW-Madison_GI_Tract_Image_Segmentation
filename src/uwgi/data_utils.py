from __future__ import annotations

import os
import re
from typing import Dict, List, Tuple

import pandas as pd


_scan_re = re.compile(r"slice_(\d+)_([0-9]+)_([0-9]+)_([0-9.]+)_([0-9.]+)\.png")


def parse_scan_filename(fname: str) -> Tuple[int, int, int]:
    """From slice_XXXX_H_W_psx_psy.png get slice index and nominal H,W in filename."""
    m = _scan_re.search(os.path.basename(fname))
    if not m:
        raise ValueError(f"Unexpected scan filename: {fname}")
    slice_idx = int(m.group(1))
    h = int(m.group(2))
    w = int(m.group(3))
    return slice_idx, h, w


def build_case_day_slices(root_dir: str) -> Dict[str, List[str]]:
    """Returns dict: key = 'case123_day0', value = sorted list of scan file paths."""
    out: Dict[str, List[str]] = {}
    for case in sorted(os.listdir(root_dir)):
        case_path = os.path.join(root_dir, case)
        if not os.path.isdir(case_path):
            continue

        for day in sorted(os.listdir(case_path)):
            day_path = os.path.join(case_path, day)
            scans_dir = os.path.join(day_path, "scans")
            if not os.path.isdir(scans_dir):
                continue

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

