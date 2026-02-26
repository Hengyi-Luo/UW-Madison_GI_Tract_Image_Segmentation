#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import cv2
except Exception as e:  # pragma: no cover
    raise RuntimeError("OpenCV (cv2) is required.") from e


def scan_dir_for_case_day(case_day: str, train_root: Path) -> Path:
    case = case_day.split("_day", 1)[0]
    return train_root / case / case_day / "scans"


def list_scan_paths(case_day: str, train_root: Path) -> list[Path]:
    d = scan_dir_for_case_day(case_day, train_root=train_root)
    if not d.is_dir():
        return []
    return sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() == ".png"])


def read_grayscale(p: Path) -> np.ndarray:
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read: {p}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def summarize_array(a: np.ndarray, *, prefix: str) -> dict[str, float]:
    a = a.astype(np.float64, copy=False)
    if a.size == 0:
        return {
            f"{prefix}_n": 0.0,
            f"{prefix}_min": math.nan,
            f"{prefix}_max": math.nan,
            f"{prefix}_mean": math.nan,
            f"{prefix}_std": math.nan,
            f"{prefix}_p01": math.nan,
            f"{prefix}_p50": math.nan,
            f"{prefix}_p99": math.nan,
        }
    return {
        f"{prefix}_n": float(a.size),
        f"{prefix}_min": float(a.min()),
        f"{prefix}_max": float(a.max()),
        f"{prefix}_mean": float(a.mean()),
        f"{prefix}_std": float(a.std(ddof=0)),
        f"{prefix}_p01": float(np.quantile(a, 0.01)),
        f"{prefix}_p50": float(np.quantile(a, 0.50)),
        f"{prefix}_p99": float(np.quantile(a, 0.99)),
    }


def sample_paths(paths: list[Path], *, n: int, seed: int) -> list[Path]:
    if not paths:
        return []
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(paths), size=min(n, len(paths)), replace=False)
    return [paths[int(i)] for i in sorted(idx.tolist())]


def main() -> None:
    p = argparse.ArgumentParser(description="Compute intensity stats for the notebook-sampled slices.")
    p.add_argument("--train_root", default="inputs/train")
    p.add_argument(
        "--selected_csv",
        default="outputs/20260206-053325_Unet3D/model_error_analysis/selected_case_days.csv",
    )
    p.add_argument(
        "--joined_csv",
        default="outputs/20260206-053325_Unet3D/model_error_analysis/eval_per_case_joined.csv",
        help="Optional: used to attach dice/cluster_id to case_day rows if present.",
    )
    p.add_argument("--n", type=int, default=6)
    p.add_argument(
        "--seeds",
        default="cluster1_worst:0,cluster1_median:1,best_cluster_median:2",
        help="Comma-separated role:seed pairs.",
    )
    p.add_argument(
        "--out_dir",
        default="outputs/20260206-053325_Unet3D/model_error_analysis",
    )
    args = p.parse_args()

    train_root = Path(args.train_root)
    selected_csv = Path(args.selected_csv)
    joined_csv = Path(args.joined_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not selected_csv.exists():
        raise SystemExit(f"Not found: {selected_csv}")

    selected = pd.read_csv(selected_csv)
    if "role" not in selected.columns or "case_day" not in selected.columns:
        raise SystemExit("selected_csv must contain 'role' and 'case_day' columns.")

    role_to_seed: dict[str, int] = {}
    for part in str(args.seeds).split(","):
        part = part.strip()
        if not part:
            continue
        role, seed_s = part.split(":", 1)
        role_to_seed[role.strip()] = int(seed_s.strip())

    joined = pd.read_csv(joined_csv) if joined_csv.exists() else None
    joined_map = None
    if joined is not None and "case_day" in joined.columns:
        keep = [c for c in ["case_day", "cluster_id", "dice_mean", "dice_large_bowel", "dice_small_bowel", "dice_stomach"] if c in joined.columns]
        joined_map = joined[keep].drop_duplicates("case_day")

    rows: list[dict[str, object]] = []
    for _, r in selected.iterrows():
        role = str(r["role"])
        case_day = str(r["case_day"])
        seed = int(role_to_seed.get(role, 0))

        all_paths = list_scan_paths(case_day, train_root=train_root)
        chosen = sample_paths(all_paths, n=args.n, seed=seed)
        if not chosen:
            rows.append({"role": role, "case_day": case_day, "error": "no_scans_found"})
            continue

        for i, pth in enumerate(chosen):
            img = read_grayscale(pth)
            flat = img.reshape(-1)
            zero_ratio = float(1.0 - (np.count_nonzero(flat) / max(int(flat.size), 1)))
            nz = flat[flat != 0]

            rec: dict[str, object] = {
                "role": role,
                "case_day": case_day,
                "seed": int(seed),
                "sample_idx": int(i),
                "png_path": str(pth),
                "H": int(img.shape[0]),
                "W": int(img.shape[1]),
                "dtype": str(img.dtype),
                "zero_ratio": zero_ratio,
            }
            rec.update(summarize_array(flat, prefix="all"))
            rec.update(summarize_array(nz, prefix="nz"))
            rows.append(rec)

    df = pd.DataFrame.from_records(rows)
    if joined_map is not None and len(df):
        df = df.merge(joined_map, on="case_day", how="left")

    out_csv = out_dir / "sampled_slice_intensity_stats.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv}")

    # Aggregate per role/case_day for quick reading
    if len(df):
        agg = (
            df.groupby(["role", "case_day"], as_index=False)
            .agg(
                n_slices=("png_path", "size"),
                zero_ratio_mean=("zero_ratio", "mean"),
                nz_mean_mean=("nz_mean", "mean"),
                nz_std_mean=("nz_std", "mean"),
                nz_p99_mean=("nz_p99", "mean"),
                nz_p50_mean=("nz_p50", "mean"),
                nz_p01_mean=("nz_p01", "mean"),
            )
        )
        out_agg = out_dir / "sampled_slice_intensity_summary_by_role.csv"
        agg.to_csv(out_agg, index=False)
        print(f"Wrote: {out_agg}")


if __name__ == "__main__":
    main()

