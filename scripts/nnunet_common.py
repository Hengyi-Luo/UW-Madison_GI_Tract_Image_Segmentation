from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import math
import os
import platform
import shlex
import shutil
import subprocess
import time
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar


REPO_ROOT = Path(__file__).resolve().parents[1]

T = TypeVar("T")


def setup_logger(log_path: Path, *, name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def load_python_module(path: Path, *, module_name: str) -> Any:
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Python module not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load Python module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_python_cfg(module: Any, path: Path) -> dict[str, Any]:
    cfg_obj = getattr(module, "cfg", None)
    if cfg_obj is None:
        raise AttributeError(f"Python config '{path}' must define a top-level 'cfg'")
    if isinstance(cfg_obj, dict):
        return dict(cfg_obj)
    if hasattr(cfg_obj, "__dict__"):
        return {k: v for k, v in vars(cfg_obj).items() if not k.startswith("_")}
    raise TypeError(f"Unsupported cfg type in '{path}': {type(cfg_obj).__name__}")


def dataclass_from_raw(cls: type[T], raw: dict[str, Any]) -> T:
    field_names = {f.name for f in fields(cls)}
    unknown = sorted(k for k in raw.keys() if k not in field_names)
    if unknown:
        raise ValueError(f"Unknown config keys for {cls.__name__}: {unknown}")
    return cls(**raw)


def cfg_to_raw_dict(cfg: Any) -> dict[str, Any]:
    if is_dataclass(cfg):
        return {f.name: getattr(cfg, f.name) for f in fields(cfg)}
    if hasattr(cfg, "__dict__"):
        return {k: v for k, v in vars(cfg).items() if not k.startswith("_")}
    raise TypeError(f"Unsupported cfg type: {type(cfg).__name__}")


def cfg_value_to_serializable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): cfg_value_to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [cfg_value_to_serializable(v) for v in value]
    return str(value)


def cfg_to_serializable_dict(cfg: Any) -> dict[str, Any]:
    return {k: cfg_value_to_serializable(v) for k, v in cfg_to_raw_dict(cfg).items()}


def save_run_config(cfg: Any, out_dir: Path, *, config_path: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(config_path)
    saved_name = f"config{source_path.suffix.lower()}" if source_path.suffix else "config.txt"
    saved_path = out_dir / saved_name
    saved_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
    (out_dir / "config.resolved.json").write_text(
        json.dumps(cfg_to_serializable_dict(cfg), indent=2) + "\n",
        encoding="utf-8",
    )
    return saved_path


def resolve_output_dir(*, output_dir: str, model_name: str) -> Path:
    if str(output_dir).strip():
        return Path(output_dir)
    ts = time.strftime("%Y%m%d-%H%M%S")
    return REPO_ROOT / "outputs" / f"{ts}_{model_name}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_git(args: list[str]) -> str:
    out = subprocess.check_output(["git", *args], cwd=str(REPO_ROOT), stderr=subprocess.STDOUT)
    return out.decode("utf-8", errors="replace").strip()


def git_status_porcelain() -> str:
    return run_git(["status", "--porcelain"])


def get_git_info() -> dict[str, str]:
    info: dict[str, str] = {}
    try:
        info["git.commit"] = run_git(["rev-parse", "HEAD"])
        info["git.branch"] = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
        info["git.describe"] = run_git(["describe", "--tags", "--always", "--dirty"])
        info["git.dirty"] = "true" if bool(git_status_porcelain().strip()) else "false"
    except Exception:
        pass
    return info


def default_mlflow_tracking_uri(cfg: Any) -> str:
    raw = cfg_to_raw_dict(cfg)
    direct = str(raw.get("mlflow_tracking_uri", "")).strip()
    if direct:
        return direct
    env_uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if env_uri:
        return env_uri
    db_path = REPO_ROOT / "mlflow.db"
    if db_path.exists():
        return f"sqlite:///{db_path}"
    return ""


def default_mlflow_experiment(cfg: Any) -> str:
    raw = cfg_to_raw_dict(cfg)
    direct = str(raw.get("mlflow_experiment", "")).strip()
    if direct:
        return direct
    env_exp = os.environ.get("MLFLOW_EXPERIMENT_NAME", "").strip()
    if env_exp:
        return env_exp
    return "uwgi/segmentation"


def write_mlflow_run_id(out_dir: Path, run_id: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mlflow_run_id.txt").write_text(str(run_id).strip() + "\n", encoding="utf-8")


def get_env_info() -> dict[str, str]:
    info: dict[str, str] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    modules = ["torch", "monai", "accelerate", "mlflow", "nnunetv2", "numpy", "nibabel"]
    for module_name in modules:
        try:
            module = __import__(module_name)
            info[module_name] = str(getattr(module, "__version__", ""))
        except Exception:
            continue
    try:
        import torch

        info["cuda.available"] = str(torch.cuda.is_available())
        info["cuda.version"] = str(torch.version.cuda) if torch.version.cuda else ""
        if torch.cuda.is_available():
            info["gpu.name"] = str(torch.cuda.get_device_name(0))
    except Exception:
        pass
    return info


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def ensure_command(command: str) -> str:
    path = shutil.which(command)
    if path is None:
        raise FileNotFoundError(f"Required command not found on PATH: {command}")
    return path


def run_command(
    cmd: list[str],
    *,
    env: dict[str, str] | None,
    logger: logging.Logger,
    cwd: Path | None = None,
) -> None:
    logger.info("$ %s", shlex.join(cmd))
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd or REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        logger.info("%s", line.rstrip())
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


def dataset_folder_name(dataset_id: int, dataset_name: str) -> str:
    return f"Dataset{int(dataset_id):03d}_{str(dataset_name).strip()}"


def make_nnunet_env(
    *,
    raw_dir: str,
    preprocessed_dir: str,
    results_dir: str,
) -> dict[str, str]:
    env = os.environ.copy()
    env["nnUNet_raw"] = str((REPO_ROOT / raw_dir).resolve())
    env["nnUNet_preprocessed"] = str((REPO_ROOT / preprocessed_dir).resolve())
    env["nnUNet_results"] = str((REPO_ROOT / results_dir).resolve())
    return env


def load_case_day_split(path: str) -> list[str]:
    rows = Path(path).read_text(encoding="utf-8").strip().splitlines()
    if not rows:
        return []
    return [line.strip() for line in rows[1:] if line.strip()]


def write_splits_final(
    preprocessed_dataset_dir: Path,
    *,
    train_ids: list[str],
    val_ids: list[str],
) -> Path:
    preprocessed_dataset_dir.mkdir(parents=True, exist_ok=True)
    path = preprocessed_dataset_dir / "splits_final.json"
    payload = [{"train": list(train_ids), "val": list(val_ids)}]
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def parse_nnunet_validation_summary(
    summary_path: Path,
    *,
    class_names: list[str],
    label_values: list[int],
) -> dict[str, float]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    parsed: dict[str, float] = {}

    def _get_metric(obj: Any, *keys: str) -> float | None:
        if not isinstance(obj, dict):
            return None
        for key in keys:
            if key in obj:
                try:
                    return float(obj[key])
                except Exception:
                    continue
        return None

    foreground_mean = payload.get("foreground_mean")
    mean_dice = _get_metric(foreground_mean, "Dice", "dice")
    if mean_dice is not None:
        parsed["official_mean_dice"] = float(mean_dice)

    mean_section = payload.get("mean")
    if isinstance(mean_section, dict):
        per_class_dice: list[float] = []
        for label_value, class_name in zip(label_values, class_names):
            entry = mean_section.get(str(label_value), mean_section.get(label_value))
            dice = _get_metric(entry, "Dice", "dice")
            if dice is not None:
                parsed[f"official_dice_{class_name}"] = float(dice)
                per_class_dice.append(float(dice))
        if "official_mean_dice" not in parsed and per_class_dice:
            parsed["official_mean_dice"] = float(sum(per_class_dice) / len(per_class_dice))

    return parsed
