#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import cv2
except Exception as e:  # pragma: no cover
    raise RuntimeError("OpenCV (cv2) is required.") from e

try:
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
except Exception as e:  # pragma: no cover
    raise RuntimeError("scikit-learn is required (sklearn).") from e


_CASE_DAY_RE = re.compile(r"^case\\d+_day\\d+$")


def iter_scan_pngs(train_dir: Path) -> Iterable[Path]:
    # Expected layout: inputs/train/case*/case*_day*/scans/*.png
    for case_dir in sorted(train_dir.iterdir()):
        if not case_dir.is_dir():
            continue
        for day_dir in sorted(case_dir.iterdir()):
            scans_dir = day_dir / "scans"
            if not scans_dir.is_dir():
                continue
            for p in scans_dir.iterdir():
                if p.is_file() and p.suffix.lower() == ".png":
                    yield p


def case_day_from_path(p: Path) -> str:
    # inputs/train/caseXXX/caseXXX_dayY/scans/file.png  ->  caseXXX_dayY
    parts = p.parts
    if len(parts) >= 3:
        candidate = parts[-3]
        if _CASE_DAY_RE.match(candidate):
            return candidate
    return "unknown"


def read_grayscale(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def downsample_array(rng: np.random.Generator, a: np.ndarray, max_n: int) -> np.ndarray:
    if a.size <= max_n:
        return a
    idx = rng.choice(a.size, size=max_n, replace=False)
    return a[idx]


def estimate_clip_from_samples(
    train_paths: list[Path],
    *,
    nonzero_only: bool,
    per_image_samples: int,
    max_total_samples: int,
    clip_low_q: float,
    clip_high_q: float,
    seed: int,
    progress_every: int,
) -> tuple[int, int]:
    rng = np.random.default_rng(seed)
    chunks: list[np.ndarray] = []
    n_samples = 0

    for i, p in enumerate(train_paths, start=1):
        img = read_grayscale(p)
        arr = img.reshape(-1)
        if nonzero_only:
            arr = arr[arr != 0]
        if arr.size:
            k = min(per_image_samples, arr.size)
            idx = rng.integers(0, arr.size, size=k, endpoint=False)
            samp = arr[idx].astype(np.uint16, copy=False)
            chunks.append(samp)
            n_samples += int(samp.size)
            if n_samples > max_total_samples * 12 // 10:
                merged = np.concatenate(chunks, axis=0)
                merged = downsample_array(rng, merged, max_total_samples)
                chunks = [merged]
                n_samples = int(merged.size)

        if progress_every and (i % progress_every == 0):
            print(f"[clip] processed {i:,}/{len(train_paths):,} images, samples={n_samples:,}")

    if n_samples == 0:
        raise RuntimeError("No samples collected; check train_dir and nonzero_only setting.")

    samples = np.concatenate(chunks, axis=0)
    samples = downsample_array(rng, samples, max_total_samples)
    lo = int(np.floor(np.quantile(samples, clip_low_q)))
    hi = int(np.ceil(np.quantile(samples, clip_high_q)))
    if hi <= lo:
        hi = lo + 1
    return lo, hi


@dataclass
class CaseDayAgg:
    n_images: int = 0
    total_pixels: int = 0
    zero_pixels: int = 0
    hist: np.ndarray | None = None  # int64[bins]

    def ensure_hist(self, bins: int) -> None:
        if self.hist is None:
            self.hist = np.zeros((bins,), dtype=np.int64)


def percentile_from_hist(edges: np.ndarray, hist: np.ndarray, q: float) -> float:
    total = float(hist.sum())
    if total <= 0:
        return 0.0
    target = q * total
    cdf = np.cumsum(hist, dtype=np.float64)
    idx = int(np.searchsorted(cdf, target, side="left"))
    idx = min(max(idx, 0), len(hist) - 1)
    lo = float(edges[idx])
    hi = float(edges[idx + 1])
    prev = float(cdf[idx - 1]) if idx > 0 else 0.0
    within = (target - prev) / max(float(hist[idx]), 1.0)
    return lo + within * (hi - lo)


def build_case_day_features(
    train_paths: list[Path],
    *,
    bins: int,
    clip_lo: int,
    clip_hi: int,
    nonzero_only_hist: bool,
    seed: int,
    progress_every: int,
) -> pd.DataFrame:
    span = int(clip_hi - clip_lo + 1)
    if span <= 1:
        raise ValueError("clip range too small")

    edges = clip_lo + (np.arange(bins + 1, dtype=np.float64) * (span / bins))
    centers = (edges[:-1] + edges[1:]) / 2.0
    centers2 = centers * centers

    aggs: dict[str, CaseDayAgg] = defaultdict(CaseDayAgg)

    for i, p in enumerate(train_paths, start=1):
        img = read_grayscale(p)
        cd = case_day_from_path(p)

        agg = aggs[cd]
        agg.n_images += 1
        agg.total_pixels += int(img.size)

        arr = img.reshape(-1)
        zero = int(arr.size - np.count_nonzero(arr))
        agg.zero_pixels += zero

        vals = arr[arr != 0] if nonzero_only_hist else arr

        if vals.size:
            vals = np.clip(vals, clip_lo, clip_hi)
            b = ((vals.astype(np.int32, copy=False) - clip_lo) * bins) // span
            b = np.minimum(np.maximum(b, 0), bins - 1)
            h = np.bincount(b, minlength=bins).astype(np.int64)
        else:
            h = np.zeros((bins,), dtype=np.int64)

        agg.ensure_hist(bins)
        agg.hist += h

        if progress_every and (i % progress_every == 0):
            print(f"[feat] processed {i:,}/{len(train_paths):,} images, case_days={len(aggs):,}")

    rows = []
    for cd, agg in sorted(aggs.items(), key=lambda x: x[0]):
        hist = agg.hist if agg.hist is not None else np.zeros((bins,), dtype=np.int64)
        nz_count = int(hist.sum())

        s = float(np.dot(hist.astype(np.float64, copy=False), centers))
        ss = float(np.dot(hist.astype(np.float64, copy=False), centers2))
        mean = (s / nz_count) if nz_count else 0.0
        var = (ss / nz_count - mean * mean) if nz_count else 0.0
        std = float(math.sqrt(max(var, 0.0)))

        p = hist.astype(np.float64) / max(float(nz_count), 1.0)
        entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum())

        zero_ratio = float(agg.zero_pixels / max(agg.total_pixels, 1))
        rows.append(
            {
                "case_day": cd,
                "n_images": int(agg.n_images),
                "total_pixels": int(agg.total_pixels),
                "zero_pixels": int(agg.zero_pixels),
                "zero_ratio": zero_ratio,
                "hist_pixels": nz_count,
                "clip_lo": int(clip_lo),
                "clip_hi": int(clip_hi),
                "bins": int(bins),
                "mean": float(mean),
                "std": float(std),
                "p01": percentile_from_hist(edges, hist, 0.01),
                "p05": percentile_from_hist(edges, hist, 0.05),
                "p50": percentile_from_hist(edges, hist, 0.50),
                "p95": percentile_from_hist(edges, hist, 0.95),
                "p99": percentile_from_hist(edges, hist, 0.99),
                "entropy": entropy,
                **{f"hist_{j:03d}": float(pj) for j, pj in enumerate(p.tolist())},
            }
        )

    return pd.DataFrame.from_records(rows)


def run_gmm_bic(
    features_df: pd.DataFrame,
    *,
    k_min: int,
    k_max: int,
    cov_types: list[str],
    pca_var: float,
    pca_max_components: int,
    n_init: int,
    reg_covar: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = [c for c in features_df.columns if c != "case_day"]
    X = features_df[feature_cols].to_numpy(dtype=np.float64, copy=True)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    n_samples, n_features = Xs.shape
    k_max = min(k_max, n_samples)
    k_min = max(1, min(k_min, k_max))

    if n_samples >= 2 and n_features >= 2 and pca_max_components > 0:
        n_components = min(int(pca_max_components), n_samples, n_features)
        pca = PCA(n_components=n_components, random_state=seed)
        Xp = pca.fit_transform(Xs)
        if 0.0 < pca_var < 1.0:
            csum = np.cumsum(pca.explained_variance_ratio_)
            keep = int(np.searchsorted(csum, pca_var, side="left")) + 1
            keep = min(keep, Xp.shape[1])
            Xp = Xp[:, :keep]
    else:
        Xp = Xs

    results = []
    best = None
    best_bic = float("inf")

    for cov in cov_types:
        for k in range(k_min, k_max + 1):
            gmm = GaussianMixture(
                n_components=k,
                covariance_type=cov,
                n_init=n_init,
                reg_covar=reg_covar,
                random_state=seed,
            )
            gmm.fit(Xp)
            bic = float(gmm.bic(Xp))
            results.append({"k": k, "covariance_type": cov, "bic": bic})
            if bic < best_bic:
                best_bic = bic
                best = gmm

    bic_df = (
        pd.DataFrame.from_records(results)
        .sort_values(["bic", "covariance_type", "k"])
        .reset_index(drop=True)
    )
    if best is None:
        raise RuntimeError("GMM sweep produced no model.")

    labels = best.predict(Xp).astype(int)
    proba = best.predict_proba(Xp).astype(np.float64)
    max_prob = proba.max(axis=1)
    second_prob = np.sort(proba, axis=1)[:, -2] if proba.shape[1] >= 2 else np.zeros_like(max_prob)

    out = pd.DataFrame(
        {
            "case_day": features_df["case_day"].astype(str).tolist(),
            "cluster_id": labels.tolist(),
            "max_prob": max_prob.tolist(),
            "second_prob": second_prob.tolist(),
        }
    )

    out.attrs["gmm"] = {
        "best_k": int(best.n_components),
        "best_covariance_type": str(best.covariance_type),
        "best_bic": float(best_bic),
        "pca_kept_dims": int(Xp.shape[1]),
        "pca_var_target": float(pca_var),
        "n_init": int(n_init),
        "reg_covar": float(reg_covar),
        "seed": int(seed),
    }
    return bic_df, out


def main() -> None:
    p = argparse.ArgumentParser(description="Case-day style clustering with GMM + BIC.")
    p.add_argument("--train_dir", default="inputs/train")
    p.add_argument("--out_dir", default="outputs/analysis")
    p.add_argument("--max_images", type=int, default=0, help="For debugging: limit number of images (0 = no limit).")
    p.add_argument("--shuffle_images", action="store_true", help="Shuffle image list before limiting.")

    p.add_argument("--bins", type=int, default=256, help="Histogram bins for nonzero intensities.")
    p.add_argument("--include_zeros_clip", action="store_true", help="Estimate clip including zeros (not recommended).")
    p.add_argument("--include_zeros_hist", action="store_true", help="Build histogram including zeros (not recommended).")
    p.add_argument("--per_image_samples", type=int, default=256, help="Pixel samples per image for clip estimation.")
    p.add_argument("--max_total_samples", type=int, default=2_000_000, help="Max pixels kept for clip estimation.")
    p.add_argument("--clip_low_q", type=float, default=0.01)
    p.add_argument("--clip_high_q", type=float, default=0.99)

    p.add_argument("--k_min", type=int, default=1)
    p.add_argument("--k_max", type=int, default=20)
    p.add_argument("--cov_types", default="full,diag,tied,spherical")
    p.add_argument("--pca_var", type=float, default=0.99, help="PCA variance target in (0,1).")
    p.add_argument("--pca_max_components", type=int, default=80)
    p.add_argument("--n_init", type=int, default=10)
    p.add_argument("--reg_covar", type=float, default=1e-6)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--progress_every", type=int, default=500)
    args = p.parse_args()

    train_dir = Path(args.train_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_paths = list(iter_scan_pngs(train_dir))
    if not train_paths:
        raise SystemExit(f"No png found under: {train_dir.resolve()}")

    if args.shuffle_images:
        rng = random.Random(args.seed)
        rng.shuffle(train_paths)
    if args.max_images and args.max_images > 0:
        train_paths = train_paths[: args.max_images]

    print(f"Found {len(train_paths):,} scan pngs.")
    print("Estimating global clip range...")
    clip_lo, clip_hi = estimate_clip_from_samples(
        train_paths,
        nonzero_only=not args.include_zeros_clip,
        per_image_samples=args.per_image_samples,
        max_total_samples=args.max_total_samples,
        clip_low_q=args.clip_low_q,
        clip_high_q=args.clip_high_q,
        seed=args.seed,
        progress_every=max(args.progress_every, 0),
    )
    print(f"clip_lo={clip_lo}, clip_hi={clip_hi}")

    print("Building case_day features...")
    feats = build_case_day_features(
        train_paths,
        bins=args.bins,
        clip_lo=clip_lo,
        clip_hi=clip_hi,
        nonzero_only_hist=not args.include_zeros_hist,
        seed=args.seed,
        progress_every=max(args.progress_every, 0),
    )

    features_csv = out_dir / "case_day_intensity_features.csv"
    feats.to_csv(features_csv, index=False)
    print(f"Wrote: {features_csv}")

    if len(feats) < 2:
        raise SystemExit(
            f"Need at least 2 case_days for GMM clustering; got {len(feats)}. "
            "Run without --max_images, or use --shuffle_images with a larger --max_images."
        )

    print("Running GMM sweep with BIC...")
    cov_types = [s.strip() for s in args.cov_types.split(",") if s.strip()]
    cluster_cols = (
        ["case_day", "zero_ratio", "mean", "std", "p01", "p05", "p50", "p95", "p99", "entropy"]
        + [c for c in feats.columns if c.startswith("hist_")]
    )
    bic_df, clusters = run_gmm_bic(
        feats[cluster_cols],
        k_min=args.k_min,
        k_max=args.k_max,
        cov_types=cov_types,
        pca_var=args.pca_var,
        pca_max_components=args.pca_max_components,
        n_init=args.n_init,
        reg_covar=args.reg_covar,
        seed=args.seed,
    )

    bic_csv = out_dir / "gmm_bic_sweep.csv"
    bic_df.to_csv(bic_csv, index=False)
    print(f"Wrote: {bic_csv}")

    clusters_csv = out_dir / "case_day_gmm_clusters.csv"
    clusters.to_csv(clusters_csv, index=False)
    print(f"Wrote: {clusters_csv}")

    meta_json = out_dir / "case_day_gmm_meta.json"
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(clusters.attrs.get("gmm", {}), f, indent=2, ensure_ascii=False)
    print(f"Wrote: {meta_json}")

    best_k = clusters.attrs["gmm"]["best_k"]
    best_cov = clusters.attrs["gmm"]["best_covariance_type"]
    best_bic = clusters.attrs["gmm"]["best_bic"]
    counts = clusters["cluster_id"].value_counts().sort_index()
    print(f"Best model: k={best_k}, cov={best_cov}, bic={best_bic:.2f}")
    print("Cluster sizes:")
    for cid, n in counts.items():
        print(f"  cluster {int(cid):2d}: {int(n):,}")


if __name__ == "__main__":
    main()

