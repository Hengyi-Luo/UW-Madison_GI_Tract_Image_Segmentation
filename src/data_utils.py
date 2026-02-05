import os
import csv
import json
import pandas as pd

def parse_scan_filename(filename: str) -> tuple[str, int, int]:
    """From slice_XXXX_H_W_psx_psy.png get slice index and H,W in filename"""
    parts = filename.split('_')
    slice_idx = int(parts[1])
    height = int(parts[2])
    width = int(parts[3])
    return filename, slice_idx, height, width

def build_case_day_slices(data_dir: str) -> dict[str, list[str]]:
    """Returns dict: key = 'case123_day0', value = sorted list of scan file paths."""
    out: dict[str, list[str]] = {}
    for case in sorted(os.listdir(data_dir)):
        case_path = os.path.join(data_dir, case)

        for day in sorted(os.listdir(case_path)):
            day_path = os.path.join(case_path, day)
            scans_dir = os.path.join(day_path, 'scans')

            scans_files = [os.path.join(scans_dir, f) for f in sorted(os.listdir(scans_dir)) if f.endswith('.png')]
        
            scans_files.sort(key=lambda f: parse_scan_filename(os.path.basename(f))[1])
            out[day] = scans_files
    return out

def build_rle_index(train_csv_path: str) -> dict[tuple[str, int], dict[str, str]]:
    """Map train.csv to idx[(case_day, slice_idx)] = {class_name: rle_string}."""
    df = pd.read_csv(train_csv_path)
    idx: dict[tuple[str, int], dict[str, str]] = {}

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

def load_case_days(ids_path: str) -> list[str]: 
    """Load case_days from csv file."""
    case_days = []
    with open(ids_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            case_days.append(row[0])
    return case_days

def load_val_vis_samples(path: str) -> list[dict]:
    with open(path, 'r') as f:
        samples = json.load(f)
    return samples