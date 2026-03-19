#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)


from src.constants import CLASSES  # noqa: E402


DEFAULT_SEEDS = (17, 42, 2026, 3407)
DEFAULT_BENCHMARK_WEIGHT = "outputs/20260318-114555_Unet3D_ref_loss_diceBCE_epoch_150/best.pt"
PYTHON = sys.executable


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def _load_python_cfg(path: Path) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location("_runtime_cfg", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = getattr(module, "cfg", None)
    if cfg is None:
        raise AttributeError(f"Missing cfg in {path}")
    if isinstance(cfg, dict):
        return dict(cfg)
    return {k: v for k, v in vars(cfg).items() if not k.startswith("_")}


def _parse_seeds(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_seed_train_config(
    *,
    base_config: Path,
    out_path: Path,
    seed: int,
    output_dir: Path,
    sw_batch_size: int,
) -> None:
    content = f"""from pathlib import Path
from types import SimpleNamespace
import importlib.util

_BASE_PATH = Path(r\"{str(base_config.resolve())}\")
_SPEC = importlib.util.spec_from_file_location(\"_base_train_cfg\", _BASE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f\"Failed to load base train config: {{_BASE_PATH}}\")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

cfg = SimpleNamespace(**{{k: v for k, v in vars(_MODULE.cfg).items() if not k.startswith(\"_\")}})
cfg.seed = {int(seed)}
cfg.output_dir = r\"{str(output_dir.resolve())}\"
cfg.sw_batch_size = {int(sw_batch_size)}

build_train_transforms = _MODULE.build_train_transforms
build_val_transforms = _MODULE.build_val_transforms
if hasattr(_MODULE, \"build_loss\"):
    build_loss = _MODULE.build_loss
"""
    _write_text(out_path, content)


def _build_infer_payload(
    *,
    base_payload: dict[str, Any],
    weights: list[str],
    output_dir: Path,
    sw_batch_size: int,
    device: str,
) -> dict[str, Any]:
    payload = dict(base_payload)
    payload["weights"] = weights[0] if len(weights) == 1 else [str(Path(w).resolve()) for w in weights]
    payload["output_dir"] = str(output_dir.resolve())
    payload["sw_batch_size"] = int(sw_batch_size)
    if device:
        payload["device"] = str(device)
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_peak_metric(*, run_dir: Path, metric_key: str, db_path: Path) -> float | None:
    run_id_path = run_dir / "mlflow_run_id.txt"
    if not run_id_path.exists() or not db_path.exists():
        return None
    run_id = run_id_path.read_text(encoding="utf-8").strip()
    if not run_id:
        return None
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.execute(
            "select max(value) from metrics where run_uuid = ? and key = ?",
            (run_id, metric_key),
        )
        row = cur.fetchone()
    if not row or row[0] is None:
        return None
    return float(row[0])


def _missed_total(summary: dict[str, Any]) -> int:
    return sum(int(summary.get(f"missed_organ_n_{cls}", 0) or 0) for cls in CLASSES)


def _spurious_total(summary: dict[str, Any]) -> int:
    return sum(int(summary.get(f"spurious_organ_n_{cls}", 0) or 0) for cls in CLASSES)


def _finite_or_inf(value: Any, *, invert: bool = False) -> float:
    try:
        out = float(value)
    except Exception:
        out = float("inf")
    if not math.isfinite(out):
        out = float("inf")
    return -out if invert else out


def _is_better_candidate(
    left: dict[str, Any],
    right: dict[str, Any] | None,
) -> bool:
    if right is None:
        return True

    left_safety = (
        _missed_total(left["summary"]),
        _spurious_total(left["summary"]),
        _finite_or_inf(left["summary"].get("hd95_mean")),
    )
    right_safety = (
        _missed_total(right["summary"]),
        _spurious_total(right["summary"]),
        _finite_or_inf(right["summary"].get("hd95_mean")),
    )
    if left_safety != right_safety:
        return left_safety < right_safety

    left_sb = float(left["summary"].get("dice_small_bowel", float("-inf")))
    right_sb = float(right["summary"].get("dice_small_bowel", float("-inf")))
    if abs(left_sb - right_sb) >= 0.002:
        return left_sb > right_sb

    left_macro = float(left["summary"].get("macro_dice", float("-inf")))
    right_macro = float(right["summary"].get("macro_dice", float("-inf")))
    if abs(left_macro - right_macro) >= 0.002:
        return left_macro > right_macro

    return int(left["num_weights"]) < int(right["num_weights"])


def _candidate_sort_key(row: dict[str, Any]) -> tuple[float, ...]:
    summary = row["summary"]
    return (
        float(_missed_total(summary)),
        float(_spurious_total(summary)),
        _finite_or_inf(summary.get("hd95_mean")),
        -float(summary.get("dice_small_bowel", float("-inf"))),
        -float(summary.get("macro_dice", float("-inf"))),
        float(row["num_weights"]),
    )


def _meets_acceptance(summary: dict[str, Any], *, benchmark: dict[str, Any] | None) -> bool:
    if float(summary.get("macro_dice", float("-inf"))) < 0.80:
        return False
    if float(summary.get("dice_small_bowel", float("-inf"))) < 0.73:
        return False
    if float(summary.get("precision_mean", float("-inf"))) < 0.78:
        return False
    if float(summary.get("recall_mean", float("-inf"))) < 0.80:
        return False
    if _missed_total(summary) != 0:
        return False
    if benchmark is not None:
        if float(summary.get("macro_dice", float("-inf"))) <= float(benchmark.get("macro_dice", float("-inf"))):
            return False
        bench_hd95 = float(benchmark.get("hd95_mean", float("inf")))
        cand_hd95 = float(summary.get("hd95_mean", float("inf")))
        if math.isfinite(bench_hd95) and not (cand_hd95 < bench_hd95):
            return False
    return True


def _candidate_specs_from_ranked(ranked_single_results: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    specs: list[tuple[str, list[dict[str, Any]]]] = []
    if ranked_single_results:
        specs.append(("best_single", ranked_single_results[:1]))
    if len(ranked_single_results) >= 2:
        specs.append(("top2_ensemble", ranked_single_results[:2]))
    if len(ranked_single_results) >= 3:
        specs.append(("top3_ensemble", ranked_single_results[:3]))
    if len(ranked_single_results) >= 4:
        specs.append(("all4_ensemble", ranked_single_results[:4]))
    return specs


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pure self-training multi-seed optimization and ensemble selection.")
    parser.add_argument("--train-config", default="configs/train_unet3d_score_long.py")
    parser.add_argument("--infer-config", default="configs/infer_unet3d_score_engineering.yaml")
    parser.add_argument("--output-root", default="outputs/selftrain_score_multiseed")
    parser.add_argument("--seeds", default="17,42,2026,3407")
    parser.add_argument("--device", default="")
    parser.add_argument("--gpu-reserved-limit-gb", type=float, default=15.2)
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--benchmark-weight", default=DEFAULT_BENCHMARK_WEIGHT)
    parser.add_argument("--stop-on-acceptance", dest="stop_on_acceptance", action="store_true")
    parser.add_argument("--no-stop-on-acceptance", dest="stop_on_acceptance", action="store_false")
    parser.set_defaults(stop_on_acceptance=True)
    args = parser.parse_args()

    base_train_config = Path(args.train_config)
    base_infer_config = Path(args.infer_config)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    runtime_train_dir = output_root / "runtime_configs" / "train"
    runtime_infer_dir = output_root / "runtime_configs" / "infer"
    db_path = REPO_ROOT / "mlflow.db"

    seeds = _parse_seeds(args.seeds)
    train_cfg_raw = _load_python_cfg(base_train_config)
    infer_cfg_raw = yaml.safe_load(base_infer_config.read_text(encoding="utf-8")) or {}

    current_train_sw_batch_size = int(train_cfg_raw.get("sw_batch_size", 2))
    current_infer_sw_batch_size = int(infer_cfg_raw.get("sw_batch_size", 2))

    benchmark_summary: dict[str, Any] | None = None
    if not bool(args.skip_benchmark):
        benchmark_weight = Path(args.benchmark_weight)
        if benchmark_weight.exists():
            benchmark_out_dir = output_root / "benchmark_existing_best"
            benchmark_cfg_path = runtime_infer_dir / "benchmark_existing_best.yaml"
            benchmark_payload = _build_infer_payload(
                base_payload=infer_cfg_raw,
                weights=[str(benchmark_weight)],
                output_dir=benchmark_out_dir,
                sw_batch_size=current_infer_sw_batch_size,
                device=str(args.device),
            )
            _write_yaml(benchmark_cfg_path, benchmark_payload)
            benchmark_summary_path = benchmark_out_dir / "eval_summary.json"
            if benchmark_summary_path.exists():
                benchmark_summary = _load_json(benchmark_summary_path)
            else:
                _run([PYTHON, "scripts/infer_unet3d.py", "--config", str(benchmark_cfg_path)])
                benchmark_summary = _load_json(benchmark_summary_path)

    seed_results: list[dict[str, Any]] = []
    candidate_cache: dict[tuple[str, ...], dict[str, Any]] = {}
    best_candidate: dict[str, Any] | None = None
    for seed in seeds:
        run_dir = output_root / f"seed_{int(seed):04d}"
        best_ckpt = run_dir / "best.pt"
        train_cfg_path = runtime_train_dir / f"train_seed_{int(seed):04d}.py"
        _write_seed_train_config(
            base_config=base_train_config,
            out_path=train_cfg_path,
            seed=int(seed),
            output_dir=run_dir,
            sw_batch_size=current_train_sw_batch_size,
        )
        if bool(args.force_retrain) or not best_ckpt.exists():
            _run([PYTHON, "scripts/train_unet3d.py", "--config", str(train_cfg_path)])

        peak_reserved_gb = _read_peak_metric(
            run_dir=run_dir,
            metric_key="system/gpu_mem_reserved_gb",
            db_path=db_path,
        )
        peak_alloc_gb = _read_peak_metric(
            run_dir=run_dir,
            metric_key="system/gpu_mem_max_allocated_gb",
            db_path=db_path,
        )
        if peak_reserved_gb is not None and float(peak_reserved_gb) > float(args.gpu_reserved_limit_gb):
            current_train_sw_batch_size = 1
            current_infer_sw_batch_size = 1

        infer_out_dir = run_dir / "infer_score"
        infer_cfg_path = runtime_infer_dir / f"infer_seed_{int(seed):04d}.yaml"
        infer_payload = _build_infer_payload(
            base_payload=infer_cfg_raw,
            weights=[str(best_ckpt)],
            output_dir=infer_out_dir,
            sw_batch_size=current_infer_sw_batch_size,
            device=str(args.device),
        )
        _write_yaml(infer_cfg_path, infer_payload)
        _run([PYTHON, "scripts/infer_unet3d.py", "--config", str(infer_cfg_path)])
        eval_summary = _load_json(infer_out_dir / "eval_summary.json")

        seed_results.append(
            {
                "name": f"seed_{int(seed):04d}",
                "seed": int(seed),
                "run_dir": str(run_dir.resolve()),
                "best_ckpt": str(best_ckpt.resolve()),
                "infer_out_dir": str(infer_out_dir.resolve()),
                "peak_reserved_gb": peak_reserved_gb,
                "peak_alloc_gb": peak_alloc_gb,
                "num_weights": 1,
                "summary": eval_summary,
                "meets_acceptance": _meets_acceptance(eval_summary, benchmark=benchmark_summary),
            }
        )
        ranked_single_results = list(sorted(seed_results, key=_candidate_sort_key))
        for base_name, rows in _candidate_specs_from_ranked(ranked_single_results):
            weights = [str(row["best_ckpt"]) for row in rows]
            cache_key = tuple(weights)
            if cache_key in candidate_cache:
                continue
            candidate_name = f"{base_name}__{'__'.join(row['name'] for row in rows)}"
            candidate_out_dir = output_root / "candidate_evals" / candidate_name
            candidate_cfg_path = runtime_infer_dir / f"{candidate_name}.yaml"
            candidate_payload = _build_infer_payload(
                base_payload=infer_cfg_raw,
                weights=weights,
                output_dir=candidate_out_dir,
                sw_batch_size=current_infer_sw_batch_size,
                device=str(args.device),
            )
            _write_yaml(candidate_cfg_path, candidate_payload)
            _run([PYTHON, "scripts/infer_unet3d.py", "--config", str(candidate_cfg_path)])
            candidate_summary = _load_json(candidate_out_dir / "eval_summary.json")
            candidate_row = {
                "name": candidate_name,
                "base_name": base_name,
                "num_weights": len(weights),
                "weights": weights,
                "infer_out_dir": str(candidate_out_dir.resolve()),
                "summary": candidate_summary,
                "meets_acceptance": _meets_acceptance(candidate_summary, benchmark=benchmark_summary),
            }
            candidate_cache[cache_key] = candidate_row
            if _is_better_candidate(candidate_row, best_candidate):
                best_candidate = candidate_row

        if bool(args.stop_on_acceptance) and best_candidate is not None and bool(best_candidate["meets_acceptance"]):
            print(
                f"Acceptance reached by {best_candidate['name']}; stopping after seed {int(seed)}.",
                flush=True,
            )
            break

    ranked_single_results = list(sorted(seed_results, key=_candidate_sort_key))
    candidate_results = [
        candidate_cache[tuple(str(row["best_ckpt"]) for row in rows)]
        for _, rows in _candidate_specs_from_ranked(ranked_single_results)
    ]

    best_candidate = None
    for row in candidate_results:
        if _is_better_candidate(row, best_candidate):
            best_candidate = row

    final_config_path: str | None = None
    if best_candidate is not None:
        final_payload = _build_infer_payload(
            base_payload=infer_cfg_raw,
            weights=[str(w) for w in best_candidate["weights"]],
            output_dir=output_root / "final_selected",
            sw_batch_size=current_infer_sw_batch_size,
            device=str(args.device),
        )
        final_config = output_root / "final_infer_config.yaml"
        _write_yaml(final_config, final_payload)
        final_config_path = str(final_config.resolve())

    seed_csv_rows = []
    for row in seed_results:
        summary = row["summary"]
        seed_csv_rows.append(
            {
                "name": row["name"],
                "seed": row["seed"],
                "macro_dice": summary.get("macro_dice"),
                "dice_small_bowel": summary.get("dice_small_bowel"),
                "precision_mean": summary.get("precision_mean"),
                "recall_mean": summary.get("recall_mean"),
                "hd95_mean": summary.get("hd95_mean"),
                "missed_total": _missed_total(summary),
                "spurious_total": _spurious_total(summary),
                "peak_reserved_gb": row["peak_reserved_gb"],
                "peak_alloc_gb": row["peak_alloc_gb"],
                "meets_acceptance": row["meets_acceptance"],
                "best_ckpt": row["best_ckpt"],
            }
        )
    candidate_csv_rows = []
    for row in candidate_results:
        summary = row["summary"]
        candidate_csv_rows.append(
            {
                "name": row["name"],
                "num_weights": row["num_weights"],
                "macro_dice": summary.get("macro_dice"),
                "dice_small_bowel": summary.get("dice_small_bowel"),
                "precision_mean": summary.get("precision_mean"),
                "recall_mean": summary.get("recall_mean"),
                "hd95_mean": summary.get("hd95_mean"),
                "missed_total": _missed_total(summary),
                "spurious_total": _spurious_total(summary),
                "meets_acceptance": row["meets_acceptance"],
                "weights": json.dumps(row["weights"]),
            }
        )

    _write_csv(output_root / "seed_metrics.csv", seed_csv_rows)
    _write_csv(output_root / "candidate_metrics.csv", candidate_csv_rows)

    summary_payload = {
        "base_train_config": str(base_train_config.resolve()),
        "base_infer_config": str(base_infer_config.resolve()),
        "output_root": str(output_root.resolve()),
        "seeds": [int(seed) for seed in seeds],
        "gpu_reserved_limit_gb": float(args.gpu_reserved_limit_gb),
        "effective_train_sw_batch_size_final": int(current_train_sw_batch_size),
        "effective_infer_sw_batch_size_final": int(current_infer_sw_batch_size),
        "benchmark_summary": benchmark_summary,
        "seed_results": seed_results,
        "candidate_results": candidate_results,
        "best_candidate": best_candidate,
        "final_infer_config": final_config_path,
    }
    _write_text(output_root / "selection_summary.json", json.dumps(summary_payload, indent=2) + "\n")

    if best_candidate is None:
        print("No candidate result was produced.", flush=True)
        return

    print(
        json.dumps(
            {
                "best_candidate": best_candidate["name"],
                "macro_dice": best_candidate["summary"].get("macro_dice"),
                "dice_small_bowel": best_candidate["summary"].get("dice_small_bowel"),
                "precision_mean": best_candidate["summary"].get("precision_mean"),
                "recall_mean": best_candidate["summary"].get("recall_mean"),
                "hd95_mean": best_candidate["summary"].get("hd95_mean"),
                "final_infer_config": final_config_path,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
