#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from scripts.eval_nnunet_baseline import evaluate_nnunet_validation  # noqa: E402
from scripts.nnunet_common import (  # noqa: E402
    cfg_to_serializable_dict,
    dataclass_from_raw,
    default_mlflow_experiment,
    default_mlflow_tracking_uri,
    ensure_command,
    extract_python_cfg,
    get_env_info,
    get_git_info,
    git_status_porcelain,
    json_safe,
    load_case_day_split,
    load_python_module,
    make_nnunet_env,
    parse_nnunet_validation_summary,
    resolve_output_dir,
    run_command,
    run_git,
    save_run_config,
    setup_logger,
    sha256_file,
    write_mlflow_run_id,
    write_splits_final,
    dataset_folder_name,
)
from scripts.prepare_nnunet_dataset import prepare_nnunet_dataset  # noqa: E402


@dataclass(frozen=True)
class TrainNNUNetCfg:
    data_root: str = "inputs"
    train_csv: str = "inputs/train.csv"
    train_ids: str = "inputs/splits/train_case_days.csv"
    val_ids: str = "inputs/splits/val_case_days.csv"

    output_dir: str = ""
    model_name: str = "nnUNet3D_fullres_baseline"

    dataset_id: int = 501
    dataset_name: str = "UWGI"
    nnunet_raw_dir: str = "nnUNet_raw"
    nnunet_preprocessed_dir: str = "nnUNet_preprocessed"
    nnunet_results_dir: str = "nnUNet_results"

    planner: str = "nnUNetPlannerResEncL"
    plans: str = "nnUNetResEncUNetPlans"
    trainer: str = "nnUNetTrainer"
    configuration: str = "3d_fullres"
    fold: int = 0
    device: str = "cuda"
    continue_training: bool = False
    verify_dataset_integrity: bool = True

    run_prepare: bool = True
    run_preprocess: bool = True
    run_post_eval: bool = True
    overwrite: bool = False
    spacing_z: float | None = None

    label_collapse_mode: str = "priority_overwrite"
    label_priority: list[str] = field(default_factory=lambda: ["large_bowel", "small_bowel", "stomach"])
    threshold: float = 0.5

    use_mlflow: bool = True
    allow_dirty_git: bool = True
    mlflow_experiment: str = "uwgi/segmentation"
    mlflow_tracking_uri: str = "sqlite:///./mlflow.db"


def _load_cfg(config_path: str) -> TrainNNUNetCfg:
    module = load_python_module(Path(config_path), module_name="train_nnunet_cfg")
    raw = extract_python_cfg(module, Path(config_path))
    return dataclass_from_raw(TrainNNUNetCfg, raw)


def _result_case_dir(cfg: TrainNNUNetCfg) -> Path:
    dataset_dir = dataset_folder_name(cfg.dataset_id, cfg.dataset_name)
    return (
        REPO_ROOT
        / cfg.nnunet_results_dir
        / dataset_dir
        / f"{cfg.trainer}__{cfg.plans}__{cfg.configuration}"
        / f"fold_{int(cfg.fold)}"
    )


def _existing_prepare_result(cfg: TrainNNUNetCfg) -> dict[str, Any]:
    dataset_dir = REPO_ROOT / cfg.nnunet_raw_dir / dataset_folder_name(cfg.dataset_id, cfg.dataset_name)
    dataset_json_path = dataset_dir / "dataset.json"
    manifest_path = dataset_dir / "manifest.csv"
    collapse_summary_path = dataset_dir / "label_collapse_summary.json"
    if not dataset_json_path.exists():
        raise FileNotFoundError(f"dataset.json not found while run_prepare=false: {dataset_json_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.csv not found while run_prepare=false: {manifest_path}")
    if not collapse_summary_path.exists():
        raise FileNotFoundError(
            f"label_collapse_summary.json not found while run_prepare=false: {collapse_summary_path}"
        )
    collapse_summary = json.loads(collapse_summary_path.read_text(encoding="utf-8"))
    return {
        "dataset_dir": dataset_dir,
        "dataset_json_path": dataset_json_path,
        "manifest_path": manifest_path,
        "collapse_summary_path": collapse_summary_path,
        "config_snapshot_path": dataset_dir / "prepare_config.resolved.json",
        "train_case_days": load_case_day_split(cfg.train_ids),
        "val_case_days": load_case_day_split(cfg.val_ids),
        "all_case_days": sorted(set(load_case_day_split(cfg.train_ids)) | set(load_case_day_split(cfg.val_ids))),
        "collapse_summary": collapse_summary,
    }


def _log_dirty_snapshot(*, out_dir: Path, mlflow) -> None:
    dirty_status = git_status_porcelain()
    if not dirty_status.strip():
        return
    mlflow.set_tag("git.dirty_allowed", "true")
    (out_dir / "git_status_porcelain.txt").write_text(dirty_status + "\n", encoding="utf-8")
    mlflow.log_artifact(str(out_dir / "git_status_porcelain.txt"), artifact_path="git")

    diff = run_git(["diff"])
    (out_dir / "git_diff.patch").write_text(diff + "\n", encoding="utf-8")
    mlflow.log_artifact(str(out_dir / "git_diff.patch"), artifact_path="git")

    diff_cached = run_git(["diff", "--cached"])
    if diff_cached.strip():
        (out_dir / "git_diff_cached.patch").write_text(diff_cached + "\n", encoding="utf-8")
        mlflow.log_artifact(str(out_dir / "git_diff_cached.patch"), artifact_path="git")


def _log_mlflow_context(
    *,
    cfg: TrainNNUNetCfg,
    args_config: str,
    out_dir: Path,
    mlflow,
) -> Any:
    dirty_status = git_status_porcelain()
    if dirty_status.strip() and not bool(cfg.allow_dirty_git):
        raise SystemExit("Refusing to run with dirty git worktree while use_mlflow=true and allow_dirty_git=false.")

    tracking_uri = default_mlflow_tracking_uri(cfg)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(default_mlflow_experiment(cfg))

    active_run = mlflow.start_run(run_name=out_dir.name)
    write_mlflow_run_id(out_dir, active_run.info.run_id)

    mlflow.set_tags(get_git_info())
    mlflow.set_tag("run_dir", str(out_dir))
    mlflow.set_tag("baseline.family", "nnunet")
    mlflow.set_tag("baseline.variant", "resenc_l")
    for key, value in get_env_info().items():
        if value != "":
            mlflow.set_tag(key, value)

    cfg_path = Path(args_config)
    if cfg_path.exists():
        mlflow.set_tag("config.path", str(cfg_path))
        mlflow.set_tag("config.sha256", sha256_file(cfg_path))
    for split_key, split_path in [("split.train", cfg.train_ids), ("split.val", cfg.val_ids)]:
        path = Path(split_path)
        if path.exists():
            mlflow.set_tag(f"{split_key}.path", str(path))
            mlflow.set_tag(f"{split_key}.sha256", sha256_file(path))

    params = cfg_to_serializable_dict(cfg)
    params.update(
        {
            "baseline.family": "nnunet",
            "baseline.variant": "resenc_l",
            "nnunet.configuration": cfg.configuration,
            "nnunet.fold": int(cfg.fold),
            "nnunet.planner": cfg.planner,
            "nnunet.plans": cfg.plans,
            "nnunet.trainer": cfg.trainer,
        }
    )
    mlflow.log_params({k: str(v) for k, v in params.items()})

    for artifact_path in [Path(cfg.train_ids), Path(cfg.val_ids)]:
        if artifact_path.exists():
            mlflow.log_artifact(str(artifact_path), artifact_path="splits")

    if dirty_status.strip() and bool(cfg.allow_dirty_git):
        _log_dirty_snapshot(out_dir=out_dir, mlflow=mlflow)

    return active_run


def _build_run_summary(
    *,
    cfg: TrainNNUNetCfg,
    nnunet_env: dict[str, str],
    prepare_result: dict[str, Any],
    result_case_dir: Path,
    official_metrics: dict[str, float],
    eval_result: dict[str, Any] | None,
    timings: dict[str, float],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "dataset_id": int(cfg.dataset_id),
        "dataset_name": str(cfg.dataset_name),
        "configuration": str(cfg.configuration),
        "planner": str(cfg.planner),
        "plans": str(cfg.plans),
        "trainer": str(cfg.trainer),
        "fold": int(cfg.fold),
        "label_collapse_mode": str(cfg.label_collapse_mode),
        "label_priority": list(cfg.label_priority),
        "nnunet_raw": str(nnunet_env["nnUNet_raw"]),
        "nnunet_preprocessed": str(nnunet_env["nnUNet_preprocessed"]),
        "nnunet_results": str(nnunet_env["nnUNet_results"]),
        "result_case_dir": str(result_case_dir),
        "checkpoint_final": str(result_case_dir / "checkpoint_final.pth"),
        "validation_summary_json": str(result_case_dir / "validation" / "summary.json"),
        "official_metrics": official_metrics,
        "prepare": {
            "dataset_json_path": str(prepare_result["dataset_json_path"]),
            "manifest_path": str(prepare_result["manifest_path"]),
            "collapse_summary_path": str(prepare_result["collapse_summary_path"]),
        },
        "timings_sec": timings,
    }
    if eval_result is not None:
        summary["repo_eval"] = {
            "eval_summary_path": str(eval_result["eval_summary_path"]),
            "eval_per_case_path": str(eval_result["eval_per_case_path"]),
            "submit_csv_path": str(eval_result["submit_csv_path"]),
            "metrics": eval_result["eval_summary_metrics"],
        }
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run nnUNet baseline with repo MLflow integration.")
    parser.add_argument("--config", type=str, default="configs/train_nnunet_baseline.py")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    cfg = _load_cfg(args.config)

    out_dir = resolve_output_dir(output_dir=cfg.output_dir, model_name=cfg.model_name)
    saved_config_path = save_run_config(cfg, out_dir, config_path=args.config)
    logger = setup_logger(out_dir / "train.log", name="train_nnunet_baseline")
    logger.info("Output directory: %s", out_dir)

    ensure_command("nnUNetv2_plan_and_preprocess")
    ensure_command("nnUNetv2_train")

    mlflow = None
    active_run = None
    if bool(cfg.use_mlflow):
        try:
            import mlflow as _mlflow  # type: ignore

            mlflow = _mlflow
        except Exception as e:
            raise RuntimeError("use_mlflow=true but mlflow import failed. Install mlflow or disable use_mlflow.") from e
        active_run = _log_mlflow_context(cfg=cfg, args_config=args.config, out_dir=out_dir, mlflow=mlflow)

    nnunet_env = make_nnunet_env(
        raw_dir=cfg.nnunet_raw_dir,
        preprocessed_dir=cfg.nnunet_preprocessed_dir,
        results_dir=cfg.nnunet_results_dir,
    )
    logger.info(
        "nnUNet paths: raw=%s preprocessed=%s results=%s",
        nnunet_env["nnUNet_raw"],
        nnunet_env["nnUNet_preprocessed"],
        nnunet_env["nnUNet_results"],
    )

    timings = {
        "prepare": 0.0,
        "preprocess": 0.0,
        "train": 0.0,
        "eval": 0.0,
    }

    try:
        prepare_started = time.time()
        if bool(cfg.run_prepare):
            prepare_result = prepare_nnunet_dataset(cfg, logger=logger)
        else:
            logger.info("run_prepare=false, reusing existing nnUNet raw dataset")
            prepare_result = _existing_prepare_result(cfg)
        timings["prepare"] = float(time.time() - prepare_started)

        preprocessed_dataset_dir = REPO_ROOT / cfg.nnunet_preprocessed_dir / dataset_folder_name(cfg.dataset_id, cfg.dataset_name)
        if bool(cfg.run_preprocess):
            preprocess_cmd = [
                "nnUNetv2_plan_and_preprocess",
                "-d",
                str(int(cfg.dataset_id)),
                "-pl",
                str(cfg.planner),
            ]
            if bool(cfg.verify_dataset_integrity):
                preprocess_cmd.append("--verify_dataset_integrity")
            preprocess_started = time.time()
            run_command(preprocess_cmd, env=nnunet_env, logger=logger)
            timings["preprocess"] = float(time.time() - preprocess_started)
        else:
            logger.info("run_preprocess=false, reusing existing nnUNet preprocessed dataset")
            plans_json = preprocessed_dataset_dir / f"{cfg.plans}.json"
            if not plans_json.exists():
                raise FileNotFoundError(f"Plans file not found while run_preprocess=false: {plans_json}")

        splits_final_path = write_splits_final(
            preprocessed_dataset_dir,
            train_ids=list(prepare_result["train_case_days"]),
            val_ids=list(prepare_result["val_case_days"]),
        )
        logger.info("Wrote nnUNet split file: %s", splits_final_path)

        train_cmd = [
            "nnUNetv2_train",
            str(int(cfg.dataset_id)),
            str(cfg.configuration),
            str(int(cfg.fold)),
            "-p",
            str(cfg.plans),
            "-device",
            str(cfg.device),
        ]
        if str(cfg.trainer).strip() and str(cfg.trainer).strip() != "nnUNetTrainer":
            train_cmd.extend(["-tr", str(cfg.trainer)])
        if bool(cfg.continue_training):
            train_cmd.append("--c")
        train_started = time.time()
        run_command(train_cmd, env=nnunet_env, logger=logger)
        timings["train"] = float(time.time() - train_started)

        result_case_dir = _result_case_dir(cfg)
        validation_summary_path = result_case_dir / "validation" / "summary.json"
        if not validation_summary_path.exists():
            raise FileNotFoundError(f"nnUNet validation summary not found after training: {validation_summary_path}")
        official_metrics = parse_nnunet_validation_summary(
            validation_summary_path,
            class_names=["large_bowel", "small_bowel", "stomach"],
            label_values=[1, 2, 3],
        )

        eval_result = None
        if bool(cfg.run_post_eval):
            eval_started = time.time()
            eval_result = evaluate_nnunet_validation(cfg, out_dir=out_dir / "eval", logger=logger)
            timings["eval"] = float(time.time() - eval_started)

        run_summary = _build_run_summary(
            cfg=cfg,
            nnunet_env=nnunet_env,
            prepare_result=prepare_result,
            result_case_dir=result_case_dir,
            official_metrics=official_metrics,
            eval_result=eval_result,
            timings=timings,
        )
        run_summary_path = out_dir / "run_summary.json"
        run_summary_path.write_text(json.dumps(json_safe(run_summary), indent=2) + "\n", encoding="utf-8")

        if mlflow is not None and active_run is not None:
            mlflow.log_metric("system/prepare_time_sec", timings["prepare"])
            mlflow.log_metric("system/preprocess_time_sec", timings["preprocess"])
            mlflow.log_metric("system/train_time_sec", timings["train"])
            mlflow.log_metric("system/eval_time_sec", timings["eval"])
            mlflow.log_metric(
                "system/total_training_time_hr",
                float(sum(timings.values())) / 3600.0,
            )

            collapse_summary = prepare_result["collapse_summary"]
            mlflow.log_metric("prepare/overlap_slices", float(collapse_summary["overlap_slices"]))
            mlflow.log_metric("prepare/overwritten_voxels", float(collapse_summary["overwritten_voxels"]))
            if "official_mean_dice" in official_metrics:
                mlflow.log_metric("val/official_mean_dice", float(official_metrics["official_mean_dice"]))
                mlflow.log_metric("summary/best_metric", float(official_metrics["official_mean_dice"]))
            for class_name in ["large_bowel", "small_bowel", "stomach"]:
                key = f"official_dice_{class_name}"
                if key in official_metrics:
                    mlflow.log_metric(f"val/official_dice_{class_name}", float(official_metrics[key]))

            if eval_result is not None:
                aux_metrics = eval_result["eval_summary_metrics"]
                if "dice_mean" in aux_metrics:
                    mlflow.log_metric("val/dice_sw_mean", float(aux_metrics["dice_mean"]))
                if "precision_mean" in aux_metrics:
                    mlflow.log_metric("val/precision_sw_mean", float(aux_metrics["precision_mean"]))
                if "recall_mean" in aux_metrics:
                    mlflow.log_metric("val/recall_sw_mean", float(aux_metrics["recall_mean"]))
                if "rve_mean" in aux_metrics:
                    mlflow.log_metric("val/rve_sw_mean", float(aux_metrics["rve_mean"]))
                for class_name in ["large_bowel", "small_bowel", "stomach"]:
                    for metric_name in ["dice", "precision", "recall", "rve"]:
                        metric_key = f"{metric_name}_{class_name}"
                        if metric_key in aux_metrics:
                            mlflow.log_metric(f"val/{metric_name}_sw_{class_name}", float(aux_metrics[metric_key]))

            mlflow.set_tag("nnunet.raw_dir", nnunet_env["nnUNet_raw"])
            mlflow.set_tag("nnunet.preprocessed_dir", nnunet_env["nnUNet_preprocessed"])
            mlflow.set_tag("nnunet.results_dir", nnunet_env["nnUNet_results"])
            mlflow.set_tag("nnunet.result_case_dir", str(result_case_dir))
            mlflow.set_tag("nnunet.validation_summary_json", str(validation_summary_path))
            mlflow.set_tag("label_collapse_mode", str(cfg.label_collapse_mode))
            mlflow.set_tag("label_priority", json.dumps(list(cfg.label_priority)))
            mlflow.set_tag("summary/stop_reason", "completed")
            mlflow.set_tag("summary/stopped_early", "false")
            if "official_mean_dice" in official_metrics:
                mlflow.set_tag("summary/best_metric_epoch", "final_validation")

            mlflow.log_artifact(str(saved_config_path), artifact_path="run")
            mlflow.log_artifact(str(out_dir / "config.resolved.json"), artifact_path="run")
            mlflow.log_artifact(str(out_dir / "train.log"), artifact_path="run")
            mlflow.log_artifact(str(run_summary_path), artifact_path="run")
            mlflow.log_artifact(str(prepare_result["dataset_json_path"]), artifact_path="dataset")
            mlflow.log_artifact(str(prepare_result["manifest_path"]), artifact_path="dataset")
            mlflow.log_artifact(str(prepare_result["collapse_summary_path"]), artifact_path="dataset")
            if Path(prepare_result["config_snapshot_path"]).exists():
                mlflow.log_artifact(str(prepare_result["config_snapshot_path"]), artifact_path="dataset")
            mlflow.log_artifact(str(splits_final_path), artifact_path="splits")
            mlflow.log_artifact(str(validation_summary_path), artifact_path="validation")
            for checkpoint_name in [
                "checkpoint_latest.pth",
                "checkpoint_final.pth",
                "checkpoint_best.pth",
            ]:
                checkpoint_path = result_case_dir / checkpoint_name
                if checkpoint_path.exists():
                    mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")
            if eval_result is not None:
                mlflow.log_artifact(str(eval_result["eval_summary_path"]), artifact_path="eval")
                mlflow.log_artifact(str(eval_result["eval_per_case_path"]), artifact_path="eval")
                mlflow.log_artifact(str(eval_result["submit_csv_path"]), artifact_path="eval")

            mlflow.end_run()

        logger.info("Finished nnUNet baseline training.")
        logger.info("Result directory     : %s", result_case_dir)
        logger.info("Validation summary   : %s", validation_summary_path)
        if bool(cfg.run_post_eval):
            logger.info("Repo eval summary    : %s", (out_dir / "eval" / "eval_summary.json"))
        logger.info("Run summary          : %s", out_dir / "run_summary.json")
    except Exception:
        if mlflow is not None and active_run is not None:
            mlflow.end_run(status="FAILED")
        raise


if __name__ == "__main__":
    main()
