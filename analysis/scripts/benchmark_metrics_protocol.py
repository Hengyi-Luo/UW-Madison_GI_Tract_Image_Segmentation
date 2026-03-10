#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.constants import CLASSES, CLASS2IDX
from src.data_utils import build_case_day_slices
from src.metrics import SegmentationMetricAccumulator
from src.rle import rle_decode


CASE_DAY_RE = re.compile(r"^(case\d+_day\d+)_slice_(\d+)$")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark protocol metrics from an inference submit.csv.")
    parser.add_argument("--run-dir", type=str, required=True, help="Inference output directory containing submit.csv.")
    parser.add_argument("--data-root", type=str, default="inputs", help="Project data root containing train/ and train.csv.")
    parser.add_argument("--ids-csv", type=str, default="", help="Split CSV listing case_day values. Defaults to <run-dir>/config.")
    parser.add_argument("--submit-csv", type=str, default="", help="Override submit CSV path. Defaults to <run-dir>/submit.csv.")
    parser.add_argument("--train-csv", type=str, default="", help="Override train CSV path. Defaults to <data-root>/train.csv.")
    parser.add_argument(
        "--compare-eval-csv",
        type=str,
        default="",
        help="Optional existing eval CSV to compare against, e.g. <run-dir>/eval_val_per_case.csv.",
    )
    parser.add_argument("--output-dir", type=str, default="", help="Output directory. Defaults to <run-dir>/metrics_protocol_benchmark.")
    return parser


def _prepare_eval_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    parsed = out["id"].astype(str).str.extract(CASE_DAY_RE)
    out["case_day"] = parsed[0]
    out["slice_idx"] = parsed[1].astype(int)
    out["segmentation"] = out["segmentation"].fillna("")
    return out


def _resolve_ids_csv(run_dir: Path, explicit_ids_csv: str) -> Path:
    if explicit_ids_csv:
        return Path(explicit_ids_csv)

    cfg_json = run_dir / "config.resolved.json"
    cfg_yaml = run_dir / "config.yaml"
    if cfg_json.exists():
        raw = json.loads(cfg_json.read_text(encoding="utf-8"))
        ids_csv = str(raw.get("ids_csv", "")).strip()
        if ids_csv:
            return Path(ids_csv)
    if cfg_yaml.exists():
        import yaml

        raw = yaml.safe_load(cfg_yaml.read_text(encoding="utf-8")) or {}
        ids_csv = str(raw.get("ids_csv", "")).strip()
        if ids_csv:
            return Path(ids_csv)
    raise FileNotFoundError("Unable to resolve ids_csv from args or run_dir config.")


def _decode_case_volume(
    case_df: pd.DataFrame,
    *,
    slice_to_z: dict[int, int],
    num_slices: int,
    height: int,
    width: int,
) -> np.ndarray:
    volume = np.zeros((len(CLASSES), num_slices, height, width), dtype=np.uint8)
    for row in case_df.to_dict("records"):
        z = slice_to_z[int(row["slice_idx"])]
        c = CLASS2IDX[str(row["class"])]
        volume[c, z] = rle_decode(str(row.get("segmentation", "")), height, width)
    return volume


def _compare_eval_csv(existing_eval_csv: Path, protocol_csv: Path, out_csv: Path) -> None:
    existing = pd.read_csv(existing_eval_csv)
    current = pd.read_csv(protocol_csv)
    merged = existing.merge(current, on=["case_day", "num_slices"], how="inner", suffixes=("_existing", "_protocol"))

    for metric_name in ["dice_mean", *[f"dice_{cls_name}" for cls_name in CLASSES]]:
        merged[f"delta_{metric_name}"] = merged[f"{metric_name}_protocol"] - merged[f"{metric_name}_existing"]

    merged.to_csv(out_csv, index=False)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def main() -> None:
    args = _build_parser().parse_args()

    run_dir = Path(args.run_dir).resolve()
    data_root = Path(args.data_root).resolve()
    submit_csv = Path(args.submit_csv).resolve() if args.submit_csv else run_dir / "submit.csv"
    train_csv = Path(args.train_csv).resolve() if args.train_csv else data_root / "train.csv"
    ids_csv = _resolve_ids_csv(run_dir, args.ids_csv).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / "metrics_protocol_benchmark"
    compare_eval_csv = Path(args.compare_eval_csv).resolve() if args.compare_eval_csv else None

    if not submit_csv.exists():
        raise FileNotFoundError(f"submit_csv not found: {submit_csv}")
    if not train_csv.exists():
        raise FileNotFoundError(f"train_csv not found: {train_csv}")
    if not ids_csv.exists():
        raise FileNotFoundError(f"ids_csv not found: {ids_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)

    pred_df = _prepare_eval_frame(pd.read_csv(submit_csv))
    gt_df = _prepare_eval_frame(pd.read_csv(train_csv, usecols=["id", "class", "segmentation"]))
    case_days = pd.read_csv(ids_csv)["case_day"].astype(str).tolist()
    case_day_slices = build_case_day_slices(str(data_root / "train"))

    pred_groups = {k: v.copy() for k, v in pred_df.groupby("case_day", sort=False)}
    gt_groups = {k: v.copy() for k, v in gt_df.groupby("case_day", sort=False)}

    acc = SegmentationMetricAccumulator(class_names=CLASSES, threshold=0.5)
    missing: list[str] = []

    for case_day in case_days:
        if case_day not in pred_groups or case_day not in gt_groups or case_day not in case_day_slices:
            missing.append(case_day)
            continue

        slice_files = case_day_slices[case_day]
        num_slices = len(slice_files)
        first_parts = Path(slice_files[0]).stem.split("_")
        height = int(first_parts[2])
        width = int(first_parts[3])

        slice_to_z: dict[int, int] = {}
        for z, path in enumerate(slice_files):
            parts = Path(path).stem.split("_")
            slice_to_z[int(parts[1])] = z

        pred_vol = _decode_case_volume(
            pred_groups[case_day],
            slice_to_z=slice_to_z,
            num_slices=num_slices,
            height=height,
            width=width,
        )
        gt_vol = _decode_case_volume(
            gt_groups[case_day],
            slice_to_z=slice_to_z,
            num_slices=num_slices,
            height=height,
            width=width,
        )

        acc.update(
            torch.from_numpy(pred_vol).unsqueeze(0).float(),
            torch.from_numpy(gt_vol).unsqueeze(0).float(),
            case_ids=[case_day],
            num_slices=[num_slices],
        )

    per_case_rows = acc.per_case_rows()
    summary = acc.summary()

    per_case_csv = output_dir / "per_case_metrics_protocol.csv"
    with per_case_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(per_case_rows[0].keys()) if per_case_rows else ["case_day", "num_slices", "dice_mean"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_case_rows)

    summary_payload: dict[str, Any] = {
        "run_dir": str(run_dir),
        "submit_csv": str(submit_csv),
        "train_csv": str(train_csv),
        "ids_csv": str(ids_csv),
        "protocol": "docs/METRICS_PROTOCOL.md",
        "hd95_units": "voxels",
        "missing_case_days": missing,
        **summary,
    }
    (output_dir / "summary_metrics_protocol.json").write_text(
        json.dumps(_json_safe(summary_payload), indent=2) + "\n",
        encoding="utf-8",
    )

    if compare_eval_csv and compare_eval_csv.exists():
        _compare_eval_csv(compare_eval_csv, per_case_csv, output_dir / "compare_existing_eval.csv")

    print(per_case_csv)
    print(output_dir / "summary_metrics_protocol.json")
    if compare_eval_csv and compare_eval_csv.exists():
        print(output_dir / "compare_existing_eval.csv")


if __name__ == "__main__":
    main()
