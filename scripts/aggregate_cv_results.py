#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    mean = float(sum(values) / len(values))
    if len(values) == 1:
        return mean, 0.0
    var = float(sum((x - mean) ** 2 for x in values) / len(values))
    return mean, math.sqrt(var)


def _load_best_checkpoint_metrics(run_dir: Path) -> dict[str, Any]:
    best_path = run_dir / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {best_path}")
    ckpt = torch.load(str(best_path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unexpected checkpoint format: {best_path}")
    metric_summary = ckpt.get("metric_summary", {})
    if not isinstance(metric_summary, dict):
        metric_summary = {}
    return {
        "best_metric": float(ckpt.get("best_metric", float("nan"))),
        "best_metric_epoch": int(ckpt.get("best_metric_epoch", -1)),
        "epoch": int(ckpt.get("epoch", -1)),
        "global_step": int(ckpt.get("global_step", -1)),
        "metric_summary": {
            str(k): (float(v) if isinstance(v, float) else int(v) if isinstance(v, int) else v)
            for k, v in metric_summary.items()
        },
    }


def _load_mlflow_runs(*, tracking_uri: str, experiment_name: str, cv_id: str) -> list[dict[str, Any]]:
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except Exception as e:  # pragma: no cover
        raise RuntimeError("aggregate_cv_results.py requires mlflow to query runs.") from e

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment not found: {experiment_name}")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        run_view_type=1,
        max_results=10000,
        order_by=["attributes.start_time ASC"],
    )
    selected: list[dict[str, Any]] = []
    for run in runs:
        tags = dict(run.data.tags)
        if str(tags.get("cv.id", "")) != str(cv_id):
            continue
        selected.append(
            {
                "run_id": str(run.info.run_id),
                "run_name": str(tags.get("mlflow.runName", "")),
                "status": str(run.info.status),
                "run_dir": str(tags.get("run_dir", "")),
                "cv_fold": int(tags.get("cv.fold", "-1") or -1),
                "cv_num_folds": int(tags.get("cv.num_folds", "0") or 0),
                "cv_scheme": str(tags.get("cv.scheme", "")),
                "cv_group_by": str(tags.get("cv.group_by", "")),
                "metrics": dict(run.data.metrics),
                "params": dict(run.data.params),
                "tags": tags,
            }
        )
    return sorted(selected, key=lambda x: (int(x["cv_fold"]), str(x["run_name"]), str(x["run_id"])))


def _build_summary(*, cv_id: str, experiment_name: str, runs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fold_rows: list[dict[str, Any]] = []
    numeric_values: dict[str, list[float]] = defaultdict(list)

    for run in runs:
        run_dir = Path(str(run["run_dir"]))
        ckpt_info = _load_best_checkpoint_metrics(run_dir)
        metric_summary = dict(ckpt_info["metric_summary"])

        row: dict[str, Any] = {
            "cv_id": str(cv_id),
            "experiment_name": str(experiment_name),
            "run_id": str(run["run_id"]),
            "run_name": str(run["run_name"]),
            "status": str(run["status"]),
            "cv_fold": int(run["cv_fold"]),
            "cv_num_folds": int(run["cv_num_folds"]),
            "cv_scheme": str(run["cv_scheme"]),
            "cv_group_by": str(run["cv_group_by"]),
            "run_dir": str(run_dir),
            "best_metric": float(ckpt_info["best_metric"]),
            "best_metric_epoch": int(ckpt_info["best_metric_epoch"]),
            "epoch": int(ckpt_info["epoch"]),
            "global_step": int(ckpt_info["global_step"]),
        }
        for key, value in metric_summary.items():
            if isinstance(value, (int, float)):
                row[key] = float(value)
                numeric_values[key].append(float(value))
        numeric_values["best_metric"].append(float(ckpt_info["best_metric"]))
        fold_rows.append(row)

    aggregate_metrics: dict[str, float] = {}
    for key, values in sorted(numeric_values.items()):
        mean, std = _mean_std(values)
        aggregate_metrics[f"{key}_mean"] = mean
        aggregate_metrics[f"{key}_std"] = std

    summary = {
        "cv_id": str(cv_id),
        "experiment_name": str(experiment_name),
        "num_folds_found": int(len(fold_rows)),
        "folds": fold_rows,
        "aggregate_metrics": aggregate_metrics,
    }
    return fold_rows, summary


def _log_summary_run(
    *,
    tracking_uri: str,
    experiment_name: str,
    cv_id: str,
    summary: dict[str, Any],
    summary_json_path: Path,
    summary_csv_path: Path,
) -> str:
    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"cv_summary_{cv_id}") as run:
        mlflow.set_tag("cv.summary", "true")
        mlflow.set_tag("cv.id", str(cv_id))
        for key, value in summary.get("aggregate_metrics", {}).items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                mlflow.log_metric(f"cv/{key}", float(value))
        mlflow.log_artifact(str(summary_json_path), artifact_path="cv")
        mlflow.log_artifact(str(summary_csv_path), artifact_path="cv")
        return str(run.info.run_id)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate K-fold results from MLflow runs using cv.id tags.")
    parser.add_argument("--cv-id", required=True, help="Cross-validation ID shared by all fold runs")
    parser.add_argument("--experiment-name", required=True, help="MLflow experiment name containing the fold runs")
    parser.add_argument("--tracking-uri", default="", help="Optional MLflow tracking URI override")
    parser.add_argument("--out-dir", default="outputs/cv_summary", help="Directory for summary CSV/JSON outputs")
    parser.add_argument(
        "--log-summary-run",
        action="store_true",
        help="Create an MLflow summary run and log aggregate metrics/artifacts there.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    runs = _load_mlflow_runs(
        tracking_uri=str(args.tracking_uri).strip(),
        experiment_name=str(args.experiment_name).strip(),
        cv_id=str(args.cv_id).strip(),
    )
    if not runs:
        raise ValueError(
            f"No MLflow runs found for cv_id={args.cv_id!r} in experiment={args.experiment_name!r}."
        )

    fold_rows, summary = _build_summary(
        cv_id=str(args.cv_id).strip(),
        experiment_name=str(args.experiment_name).strip(),
        runs=runs,
    )

    out_dir = Path(args.out_dir) / str(args.cv_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv_path = out_dir / "fold_metrics.csv"
    summary_json_path = out_dir / "summary.json"

    fieldnames: list[str] = []
    for row in fold_rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(summary_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(fold_rows)
    summary_json_path.write_text(json.dumps(_json_safe(summary), indent=2) + "\n", encoding="utf-8")

    if bool(args.log_summary_run):
        summary_run_id = _log_summary_run(
            tracking_uri=str(args.tracking_uri).strip(),
            experiment_name=str(args.experiment_name).strip(),
            cv_id=str(args.cv_id).strip(),
            summary=summary,
            summary_json_path=summary_json_path,
            summary_csv_path=summary_csv_path,
        )
        summary["summary_run_id"] = str(summary_run_id)
        summary_json_path.write_text(json.dumps(_json_safe(summary), indent=2) + "\n", encoding="utf-8")
        print(f"Logged MLflow summary run: {summary_run_id}")

    print(f"Wrote fold metrics: {summary_csv_path}")
    print(f"Wrote summary: {summary_json_path}")


if __name__ == "__main__":
    main()
