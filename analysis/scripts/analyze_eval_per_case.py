#!/usr/bin/env python3
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd


def _maybe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def build_case_day_label_features(meta_csv: Path) -> pd.DataFrame:
    usecols = [
        "case_day",
        "class",
        "label_area",
        "label_frac",
        "label_positive",
    ]
    meta = pd.read_csv(meta_csv, usecols=usecols)

    g = meta.groupby(["case_day", "class"], as_index=False)
    by = g.agg(
        n_slices=("label_positive", "size"),
        pos_slices=("label_positive", "sum"),
        pos_rate=("label_positive", "mean"),
        sum_area=("label_area", "sum"),
        mean_area=("label_area", "mean"),
        mean_frac=("label_frac", "mean"),
    )

    # Positive-only stats
    pos = meta.loc[meta["label_positive"]].copy()
    if len(pos) == 0:
        by["pos_mean_area"] = 0.0
        by["pos_mean_frac"] = 0.0
    else:
        gp = pos.groupby(["case_day", "class"], as_index=False).agg(
            pos_mean_area=("label_area", "mean"),
            pos_mean_frac=("label_frac", "mean"),
        )
        by = by.merge(gp, on=["case_day", "class"], how="left")
        by[["pos_mean_area", "pos_mean_frac"]] = by[["pos_mean_area", "pos_mean_frac"]].fillna(0.0)

    # Wide format: one row per case_day
    wide = by.pivot(index="case_day", columns="class")
    wide.columns = [f"label_{stat}_{cls}" for stat, cls in wide.columns]
    wide = wide.reset_index()
    return wide


def build_case_day_slice_level_features(per_slice_csv: Path) -> pd.DataFrame | None:
    if not per_slice_csv.exists():
        return None
    usecols = ["case_day", "any_positive", "total_area", "total_frac", "H", "W"]
    ps = pd.read_csv(per_slice_csv, usecols=usecols)

    def _mode_int(s: pd.Series) -> int:
        m = s.mode()
        if len(m):
            return int(m.iloc[0])
        return int(s.iloc[0])

    g = ps.groupby("case_day", as_index=False).agg(
        slices=("any_positive", "size"),
        any_pos_slices=("any_positive", "sum"),
        any_pos_rate=("any_positive", "mean"),
        total_area_sum=("total_area", "sum"),
        total_area_mean=("total_area", "mean"),
        total_frac_mean=("total_frac", "mean"),
        H_mode=("H", _mode_int),
        W_mode=("W", _mode_int),
    )
    g["any_pos_slices"] = g["any_pos_slices"].astype(int)
    return g


def corr_table(df: pd.DataFrame, target_cols: list[str], feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for t in target_cols:
        for f in feature_cols:
            a = df[t].to_numpy(dtype=float, copy=False)
            b = df[f].to_numpy(dtype=float, copy=False)
            mask = np.isfinite(a) & np.isfinite(b)
            if mask.sum() < 3:
                r = np.nan
            else:
                r = float(np.corrcoef(a[mask], b[mask])[0, 1])
            rows.append({"target": t, "feature": f, "pearson_r": r, "n": int(mask.sum())})
    out = pd.DataFrame.from_records(rows)
    out["abs_r"] = out["pearson_r"].abs()
    return out.sort_values(["target", "abs_r"], ascending=[True, False]).drop(columns=["abs_r"])


def _md_escape(s: object) -> str:
    return str(s).replace("|", "\\|")


def _md_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df is None or len(df) == 0:
        return "_(empty)_\n"
    df2 = df.head(max_rows).copy()
    cols = list(df2.columns)
    lines = []
    lines.append("| " + " | ".join(_md_escape(c) for c in cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df2.iterrows():
        lines.append("| " + " | ".join(_md_escape(row[c]) for c in cols) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_(showing first {max_rows} of {len(df)})_\n")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze eval_per_case.csv with label/intensity/clustering metadata.")
    p.add_argument("--eval_csv", required=True)
    p.add_argument("--meta_csv", default="outputs/analysis/meta_info_table.csv")
    p.add_argument("--per_slice_csv", default="outputs/analysis/00_label_analysis/label_imbalance_per_slice.csv")
    p.add_argument("--intensity_features_csv", default="outputs/analysis/case_day_intensity_features.csv")
    p.add_argument("--clusters_csv", default="outputs/analysis/case_day_gmm_clusters.csv")
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    eval_csv = Path(args.eval_csv)
    meta_csv = Path(args.meta_csv)
    per_slice_csv = Path(args.per_slice_csv)
    intensity_csv = Path(args.intensity_features_csv)
    clusters_csv = Path(args.clusters_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not eval_csv.exists():
        raise SystemExit(f"Not found: {eval_csv}")
    if not meta_csv.exists():
        raise SystemExit(f"Not found: {meta_csv}")

    eval_df = pd.read_csv(eval_csv)
    if "case_day" not in eval_df.columns:
        raise SystemExit("eval_csv must contain a 'case_day' column.")

    label_wide = build_case_day_label_features(meta_csv)
    slice_feat = build_case_day_slice_level_features(per_slice_csv)

    merged = eval_df.merge(label_wide, on="case_day", how="left")
    if slice_feat is not None:
        merged = merged.merge(slice_feat, on="case_day", how="left")

    intensity = _maybe_read_csv(intensity_csv)
    if intensity is not None and "case_day" in intensity.columns:
        merged = merged.merge(intensity, on="case_day", how="left", suffixes=("", "_intensity"))

    clusters = _maybe_read_csv(clusters_csv)
    if clusters is not None and "case_day" in clusters.columns:
        merged = merged.merge(clusters, on="case_day", how="left", suffixes=("", "_cluster"))

    out_joined = out_dir / "eval_per_case_joined.csv"
    merged.to_csv(out_joined, index=False)

    # Worst cases
    worst_mean = merged.sort_values("dice_mean", ascending=True).head(20)[
        ["case_day", "dice_mean", "dice_large_bowel", "dice_small_bowel", "dice_stomach", "num_slices"]
    ]
    worst_lb = merged.sort_values("dice_large_bowel", ascending=True).head(20)[["case_day", "dice_large_bowel"]]
    worst_sb = merged.sort_values("dice_small_bowel", ascending=True).head(20)[["case_day", "dice_small_bowel"]]
    worst_st = merged.sort_values("dice_stomach", ascending=True).head(20)[["case_day", "dice_stomach"]]

    target_cols = [c for c in ["dice_mean", "dice_large_bowel", "dice_small_bowel", "dice_stomach"] if c in merged.columns]
    feat_candidates = [
        # slice-level label
        "any_pos_rate",
        "total_area_sum",
        "total_frac_mean",
        # per-class label
        "label_pos_rate_large_bowel",
        "label_pos_rate_small_bowel",
        "label_pos_rate_stomach",
        "label_pos_mean_area_large_bowel",
        "label_pos_mean_area_small_bowel",
        "label_pos_mean_area_stomach",
        # intensity
        "zero_ratio",
        "mean",
        "std",
        "p01",
        "p50",
        "p99",
        "entropy",
        # clustering
        "max_prob",
        "second_prob",
    ]
    feature_cols = [c for c in feat_candidates if c in merged.columns]
    corrs = corr_table(merged, target_cols, feature_cols) if (target_cols and feature_cols) else pd.DataFrame()

    corr_csv = out_dir / "pearson_correlations.csv"
    if len(corrs):
        corrs.to_csv(corr_csv, index=False)

    cluster_summary = None
    if "cluster_id" in merged.columns:
        cluster_summary = (
            merged.groupby("cluster_id", as_index=False)
            .agg(
                n=("case_day", "size"),
                dice_mean_mean=("dice_mean", "mean"),
                dice_mean_median=("dice_mean", "median"),
                dice_mean_p10=("dice_mean", lambda x: float(np.quantile(x, 0.10))),
            )
            .sort_values("dice_mean_mean", ascending=True)
        )
        cluster_summary.to_csv(out_dir / "dice_by_cluster.csv", index=False)

    # Plots
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        ax.hist(merged["dice_mean"].to_numpy(dtype=float), bins=30)
        ax.set_title("dice_mean distribution (per case_day)")
        ax.set_xlabel("dice_mean")
        ax.set_ylabel("count")
        fig.savefig(out_dir / "dice_mean_hist.png", dpi=200)
        plt.close(fig)

        if "any_pos_rate" in merged.columns:
            fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
            ax.scatter(merged["any_pos_rate"], merged["dice_mean"], s=12, alpha=0.7)
            ax.set_xlabel("any_pos_rate (slice-level)")
            ax.set_ylabel("dice_mean")
            ax.set_title("dice_mean vs any_pos_rate")
            fig.savefig(out_dir / "scatter_dice_mean_vs_any_pos_rate.png", dpi=200)
            plt.close(fig)

        if "cluster_id" in merged.columns:
            order = merged.groupby("cluster_id")["dice_mean"].mean().sort_values(ascending=True).index.tolist()
            data = [merged.loc[merged["cluster_id"] == cid, "dice_mean"].to_numpy(dtype=float) for cid in order]
            fig, ax = plt.subplots(figsize=(max(8, len(order) * 0.5), 4), constrained_layout=True)
            ax.boxplot(data, tick_labels=[str(int(c)) for c in order], showfliers=False)
            ax.set_xlabel("cluster_id (sorted by mean dice)")
            ax.set_ylabel("dice_mean")
            ax.set_title("dice_mean by style cluster (GMM)")
            fig.savefig(out_dir / "boxplot_dice_mean_by_cluster.png", dpi=200)
            plt.close(fig)

    except Exception as e:
        (out_dir / "plot_errors.txt").write_text(str(e) + "\n", encoding="utf-8")

    # Markdown summary
    missing_labels = int(merged.filter(regex=r"^label_").isna().any(axis=1).sum()) if any(
        c.startswith("label_") for c in merged.columns
    ) else 0
    missing_intensity = int(merged["mean"].isna().sum()) if "mean" in merged.columns else None
    missing_cluster = int(merged["cluster_id"].isna().sum()) if "cluster_id" in merged.columns else None

    parts = []
    parts.append("# Model error analysis (eval_per_case)\n")
    parts.append("## Inputs\n")
    parts.append(f"- eval_csv: `{eval_csv}`\n")
    parts.append(f"- meta_csv: `{meta_csv}`\n")
    parts.append(f"- per_slice_csv: `{per_slice_csv}`\n")
    parts.append(f"- intensity_features_csv: `{intensity_csv}` (optional)\n")
    parts.append(f"- clusters_csv: `{clusters_csv}` (optional)\n\n")

    parts.append("## Merge coverage\n")
    parts.append(f"- eval rows: {len(eval_df):,}\n")
    parts.append(f"- missing label features (any label_* NA in row): {missing_labels:,}\n")
    if missing_intensity is not None:
        parts.append(f"- missing intensity features (mean is NA): {missing_intensity:,}\n")
    if missing_cluster is not None:
        parts.append(f"- missing cluster assignment (cluster_id is NA): {missing_cluster:,}\n")
    parts.append("\n")

    parts.append("## Worst case_days (by dice_mean)\n")
    parts.append(_md_table(worst_mean))
    parts.append("\n## Worst case_days (by dice_large_bowel)\n")
    parts.append(_md_table(worst_lb))
    parts.append("\n## Worst case_days (by dice_small_bowel)\n")
    parts.append(_md_table(worst_sb))
    parts.append("\n## Worst case_days (by dice_stomach)\n")
    parts.append(_md_table(worst_st))

    if len(corrs):
        parts.append("\n## Pearson correlations (top 8 per target)\n\n")
        for t in target_cols:
            top = corrs.loc[corrs["target"] == t].head(8)
            parts.append(f"### {t}\n")
            parts.append(_md_table(top))
        parts.append(f"\nFull table: `{corr_csv}`\n")

    if cluster_summary is not None:
        parts.append("\n## Cluster summary (sorted by mean dice_mean)\n")
        parts.append(_md_table(cluster_summary, max_rows=50))
        parts.append(f"\nFull table: `{out_dir / 'dice_by_cluster.csv'}`\n")

    summary_md = out_dir / "summary.md"
    summary_md.write_text("".join(parts), encoding="utf-8")

    print(f"Wrote: {out_joined}")
    print(f"Wrote: {summary_md}")
    if len(corrs):
        print(f"Wrote: {corr_csv}")

    if intensity is None:
        print(
            textwrap.dedent(
                f"""
                NOTE: intensity features not found at {intensity_csv}.
                To generate intensity features + clustering, run:
                  python analysis/scripts/cluster_case_day_gmm_bic.py --train_dir inputs/train --out_dir outputs/analysis
                """
            ).strip()
        )


if __name__ == "__main__":
    main()

