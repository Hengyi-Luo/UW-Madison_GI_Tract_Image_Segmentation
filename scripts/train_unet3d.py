#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)


import argparse
import hashlib
import importlib.util
import json
import logging
import math
import platform
import subprocess
import time
from dataclasses import dataclass, field, fields
from typing import Any, Sequence

import torch
from accelerate import Accelerator
from monai.data import CacheDataset, DataLoader
from monai.data import pad_list_data_collate
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, DiceFocalLoss, DiceLoss, FocalLoss
from monai.networks.layers import Norm
from monai.networks.nets import Unet
from monai.utils import set_determinism
from torch.utils.tensorboard import SummaryWriter

from src.constants import CLASSES
from src.data_utils import build_case_day_slices, build_rle_index, load_case_days, load_val_vis_samples
from src.losses import DiceBceMultilabelLoss
from src.metrics import DEFAULT_METRIC_NAMES, SegmentationMetricAccumulator

try:
    import pynvml  # type: ignore
except Exception:  # pragma: no cover
    pynvml = None


@dataclass(frozen=True)
class TrainCfg:
    # data
    data_root: str = "inputs"
    train_ids: str = "inputs/splits/train_case_days.csv"
    val_ids: str = "inputs/splits/val_case_days.csv"
    num_workers: int = 6
    seed: int = 42

    # checkpointing
    output_dir: str = ""
    resume_from: str = ""

    # cross-validation metadata
    cv_enabled: bool = False
    cv_id: str = ""
    cv_fold: int = -1
    cv_num_folds: int = 0
    cv_scheme: str = ""
    cv_group_by: str = ""

    # tracking
    # If true: enforce clean git worktree and log params/metrics/artifacts to MLflow.
    use_mlflow: bool = False
    # When use_mlflow is true:
    # - allow_dirty_git=false: refuse to start if git worktree is dirty
    # - allow_dirty_git=true: proceed and (best-effort) log git diff/status to MLflow artifacts
    allow_dirty_git: bool = False
    # If empty: use $MLFLOW_EXPERIMENT_NAME or a repo-default.
    mlflow_experiment: str = ""
    # If empty: use sqlite:///<repo>/mlflow.db when present, otherwise MLflow default.
    mlflow_tracking_uri: str = ""

    # model
    model_name: str = "Unet3D_IMREAD_UNCHANGED"
    spatial_dims: int = 3
    in_channels: int = 1
    out_channels: int = len(CLASSES)

    # patches
    patch_d: int = 96
    patch_h: int = 224
    patch_w: int = 224

    # optimizer
    lr: float = 1e-4
    weight_decay: float = 1e-4

    # loss
    loss_name: str = "dicece"  # dicece | dicebce | dice | dicefocal | focal
    loss_squared_pred: bool = True
    loss_reduction: str = "mean"
    loss_lambda_dice: float = 1.0
    loss_lambda_ce: float = 1.0
    loss_lambda_focal: float = 1.0
    loss_gamma: float = 2.0
    loss_alpha: float | None = None
    loss_label_smoothing: float = 0.0
    loss: Any | None = field(default=None, repr=False, compare=False)

    # OneCycleLR
    lr_max: float = 5e-4
    lr_pct_start: float = 0.2
    lr_div_factor: float = 25.0
    lr_final_div_factor: float = 1000.0

    # training
    epochs: int = 150
    val_interval: int = 3
    train_patch_metrics_interval: int = 0
    val_hd95_interval: int = 0
    mixed_precision: str = "fp16"  # no | fp16 | bf16
    train_batch_size: int = 16
    val_batch_size: int = 1

    # val (sliding window)
    sw_batch_size: int = 4
    sw_overlap: float = 0.25

    # visualization samples metadata (loaded but not used by this script yet)
    val_vis_every: int = 5
    val_vis_samples: str = "inputs/splits/val_vis_samples.json"

    # caching
    train_cache_rate: float = 1.0
    val_cache_rate: float = 1.0

    # early stopping
    early_stop_enabled: bool = False
    early_stop_metric: str = "val/dice_sw_mean"
    early_stop_mode: str = "max"  # max | min
    early_stop_min_epochs: int = 0
    early_stop_patience: int = 0
    early_stop_min_delta: float = 0.0

    @property
    def patch_size(self) -> tuple[int, int, int]:
        return (int(self.patch_d), int(self.patch_h), int(self.patch_w))


def _setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def _load_cfg(path: str) -> tuple[TrainCfg, Any]:
    cfg_path = Path(path)
    if cfg_path.suffix != ".py":
        raise ValueError(f"Config must be a Python module (.py), got: {cfg_path}")
    module = _load_python_module(cfg_path, module_name="train_unet3d_config")
    raw = _extract_python_cfg(module, cfg_path)

    known = set(TrainCfg().__dict__.keys())
    unknown = sorted([k for k in raw.keys() if k not in known])
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")

    merged = {**TrainCfg().__dict__, **raw}
    return TrainCfg(**merged), module


def _resolve_output_dir(cfg: TrainCfg) -> Path:
    if cfg.output_dir:
        return Path(cfg.output_dir)
    ts = time.strftime("%Y%m%d-%H%M%S")
    return Path("outputs") / f"{ts}_{cfg.model_name}"


def _cv_enabled(cfg: TrainCfg) -> bool:
    return bool(cfg.cv_enabled) or bool(str(cfg.cv_id).strip()) or int(cfg.cv_num_folds) > 0 or int(cfg.cv_fold) >= 0


def _extract_python_cfg(module: Any, path: Path) -> dict[str, Any]:
    cfg_obj = getattr(module, "cfg", None)
    if cfg_obj is None:
        raise AttributeError(f"Python config '{path}' must define a top-level 'cfg'")
    if isinstance(cfg_obj, dict):
        return dict(cfg_obj)
    if hasattr(cfg_obj, "__dict__"):
        return {k: v for k, v in vars(cfg_obj).items() if not k.startswith("_")}
    raise TypeError(f"Unsupported cfg type in '{path}': {type(cfg_obj).__name__}")


def _cfg_to_raw_dict(cfg: TrainCfg) -> dict[str, Any]:
    return {f.name: getattr(cfg, f.name) for f in fields(cfg)}


def _cfg_value_to_serializable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.nn.Module):
        return value.__class__.__name__
    if isinstance(value, dict):
        return {str(k): _cfg_value_to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_cfg_value_to_serializable(v) for v in value]
    return str(value)


def _cfg_to_serializable_dict(cfg: TrainCfg) -> dict[str, Any]:
    return {k: _cfg_value_to_serializable(v) for k, v in _cfg_to_raw_dict(cfg).items()}


def _save_run_config(cfg: TrainCfg, out_dir: Path, *, config_path: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(config_path)
    saved_name = f"config{source_path.suffix.lower()}" if source_path.suffix else "config.txt"
    saved_path = out_dir / saved_name
    saved_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
    (out_dir / "config.resolved.json").write_text(json.dumps(_cfg_to_serializable_dict(cfg), indent=2), encoding="utf-8")
    return saved_path


def _write_resolved_runtime_config(out_dir: Path, cfg: TrainCfg, *, steps_per_epoch: int) -> dict[str, int | float | str]:
    payload: dict[str, Any] = dict(_cfg_to_serializable_dict(cfg))
    payload["steps_per_epoch"] = int(steps_per_epoch)
    payload["total_steps"] = int(steps_per_epoch) * int(cfg.epochs)
    payload["val_interval_steps"] = int(steps_per_epoch) * int(cfg.val_interval)
    payload["val_vis_every_steps"] = int(steps_per_epoch) * int(cfg.val_vis_every)
    (out_dir / "config.resolved.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _build_transforms(*, cfg: TrainCfg, cfg_module: Any, case_day_slices, rle_index):
    module = cfg_module
    build_train_transforms = getattr(module, "build_train_transforms", None)
    build_val_transforms = getattr(module, "build_val_transforms", None)
    if build_train_transforms is None or build_val_transforms is None:
        raise AttributeError(
            "Config module must define 'build_train_transforms' and 'build_val_transforms'"
        )
    train_transforms = build_train_transforms(
        cfg=cfg,
        case_day_slices=case_day_slices,
        rle_index=rle_index,
    )
    val_transforms = build_val_transforms(
        cfg=cfg,
        case_day_slices=case_day_slices,
        rle_index=rle_index,
    )
    return train_transforms, val_transforms


def _load_python_module(path: Path, *, module_name: str):
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


def _pad_collate_keep_slice_idxs(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad-collate tensors but keep non-tensor volume metadata as Python lists."""
    metadata_keys = ("slice_idxs", "orig_shape")
    preserved = {key: [item.get(key) for item in batch] for key in metadata_keys}
    batch_no_idxs: list[dict[str, Any]] = []
    for item in batch:
        item = dict(item)
        for key in metadata_keys:
            item.pop(key, None)
        batch_no_idxs.append(item)
    collated = pad_list_data_collate(batch_no_idxs)
    collated.update(preserved)
    return collated


def _read_gpu_stats() -> dict[str, float]:
    """Return GPU utilization and memory stats via NVML, or empty if unavailable."""
    if pynvml is None:
        return {}
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gb = 1024.0 * 1024.0 * 1024.0
        return {
            "gpu_utilization_pct": float(util.gpu),
            "gpu_mem_utilization_pct": float(util.memory),
            "gpu_mem_used_gb": float(mem.used) / gb,
        }
    except Exception:
        return {}
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_git(args: list[str]) -> str:
    out = subprocess.check_output(["git", *args], cwd=str(REPO_ROOT), stderr=subprocess.STDOUT)
    return out.decode("utf-8", errors="replace").strip()


def _git_status_porcelain() -> str:
    return _run_git(["status", "--porcelain"])


def _get_git_info() -> dict[str, str]:
    info: dict[str, str] = {}
    try:
        info["git.commit"] = _run_git(["rev-parse", "HEAD"])
        info["git.branch"] = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
        info["git.describe"] = _run_git(["describe", "--tags", "--always", "--dirty"])
        info["git.dirty"] = "true" if bool(_git_status_porcelain().strip()) else "false"
    except Exception:
        # Repo may not be a git checkout in some environments.
        pass
    return info


def _default_mlflow_tracking_uri(cfg: TrainCfg) -> str:
    if cfg.mlflow_tracking_uri:
        return str(cfg.mlflow_tracking_uri)
    env_uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if env_uri:
        return env_uri
    db_path = REPO_ROOT / "mlflow.db"
    if db_path.exists():
        return f"sqlite:///{db_path}"
    return ""


def _default_mlflow_experiment(cfg: TrainCfg) -> str:
    if cfg.mlflow_experiment:
        return str(cfg.mlflow_experiment)
    env_exp = os.environ.get("MLFLOW_EXPERIMENT_NAME", "").strip()
    if env_exp:
        return env_exp
    return "uwgi/segmentation"


def _load_mlflow_run_id_from_resume(resume_from: str) -> str:
    if not resume_from:
        return ""
    p = Path(resume_from)
    run_id_path = p.parent / "mlflow_run_id.txt"
    if run_id_path.exists():
        return run_id_path.read_text(encoding="utf-8").strip()
    return ""


def _write_mlflow_run_id(out_dir: Path, run_id: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mlflow_run_id.txt").write_text(str(run_id).strip() + "\n", encoding="utf-8")


def _get_env_info() -> dict[str, str]:
    info: dict[str, str] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        info["torch"] = str(torch.__version__)
        info["cuda.available"] = str(torch.cuda.is_available())
        info["cuda.version"] = str(torch.version.cuda) if torch.version.cuda else ""
        info["cudnn.version"] = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else ""
        if torch.cuda.is_available():
            info["gpu.name"] = str(torch.cuda.get_device_name(0))
    except Exception:
        pass
    try:
        import monai  # type: ignore

        info["monai"] = str(monai.__version__)
    except Exception:
        pass
    try:
        import accelerate  # type: ignore

        info["accelerate"] = str(accelerate.__version__)
    except Exception:
        pass
    return info


def _build_model(cfg: TrainCfg) -> torch.nn.Module:
    return Unet(
        spatial_dims=int(cfg.spatial_dims),
        in_channels=int(cfg.in_channels),
        out_channels=int(cfg.out_channels),
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=0.2,
        norm=Norm.BATCH,
    )


def _build_loss(*, cfg: TrainCfg, cfg_module: Any) -> torch.nn.Module:
    if cfg.loss is not None:
        if not isinstance(cfg.loss, torch.nn.Module):
            raise TypeError(f"cfg.loss must be torch.nn.Module, got {type(cfg.loss).__name__}")
        return cfg.loss

    build_loss = getattr(cfg_module, "build_loss", None)
    if build_loss is not None:
        loss_function = build_loss(cfg=cfg)
        if not isinstance(loss_function, torch.nn.Module):
            raise TypeError(f"build_loss(cfg=...) must return torch.nn.Module, got {type(loss_function).__name__}")
        return loss_function

    loss_name = str(cfg.loss_name).strip().lower()
    common_kwargs = {
        "sigmoid": True,
        "squared_pred": bool(cfg.loss_squared_pred),
        "reduction": str(cfg.loss_reduction),
    }
    alpha = None if cfg.loss_alpha is None else float(cfg.loss_alpha)

    if loss_name == "dicece":
        return DiceCELoss(
            **common_kwargs,
            lambda_dice=float(cfg.loss_lambda_dice),
            lambda_ce=float(cfg.loss_lambda_ce),
            label_smoothing=float(cfg.loss_label_smoothing),
        )
    if loss_name == "dicebce":
        return DiceBceMultilabelLoss(
            w_dice=float(cfg.loss_lambda_dice),
            w_bce=float(cfg.loss_lambda_ce),
            reduction=str(cfg.loss_reduction),
        )
    if loss_name == "dice":
        return DiceLoss(**common_kwargs)
    if loss_name == "dicefocal":
        return DiceFocalLoss(
            **common_kwargs,
            gamma=float(cfg.loss_gamma),
            lambda_dice=float(cfg.loss_lambda_dice),
            lambda_focal=float(cfg.loss_lambda_focal),
            alpha=alpha,
        )
    if loss_name == "focal":
        return FocalLoss(
            to_onehot_y=False,
            use_softmax=False,
            gamma=float(cfg.loss_gamma),
            alpha=alpha,
            reduction=str(cfg.loss_reduction),
        )
    raise ValueError(f"Unsupported loss_name: {cfg.loss_name}")


def _load_resume(
    cfg: TrainCfg,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    logger: logging.Logger,
) -> tuple[int, int, float, int]:
    if not cfg.resume_from:
        return 1, 0, -1.0, -1

    p = Path(cfg.resume_from)
    if not p.exists():
        raise FileNotFoundError(f"resume_from not found: {p}")

    ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError("resume_from must be a dict checkpoint with 'model_state_dict'.")

    model.load_state_dict(ckpt["model_state_dict"])
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    start_epoch = int(ckpt.get("epoch", 1))
    global_step = int(ckpt.get("global_step", 0))
    best_metric = float(ckpt.get("best_metric", -1.0))
    best_metric_epoch = int(ckpt.get("best_metric_epoch", -1))
    logger.info(
        "Resumed from %s (epoch=%d, global_step=%d, best=%.4f@%d)",
        str(p),
        start_epoch,
        global_step,
        best_metric,
        best_metric_epoch,
    )
    return start_epoch, global_step, best_metric, best_metric_epoch


def _batch_to_device(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    images = batch["image"].to(device)
    labels = batch["label"].to(device).float()
    return images, labels


def _to_case_day(x: Any) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)) and x and isinstance(x[0], str):
        return x[0]
    if torch.is_tensor(x) and x.numel() == 1:
        try:
            return str(x.item())
        except Exception:
            return str(x)
    return str(x)


def _extract_batch_orig_shape(batch: dict[str, Any], *, batch_index: int = 0) -> tuple[int, int, int] | None:
    orig_shape = batch.get("orig_shape")
    if orig_shape is None:
        return None

    def _to_int(item: Any) -> int:
        if torch.is_tensor(item):
            if item.ndim == 0:
                return int(item.item())
            return int(item[int(batch_index)].item())
        if isinstance(item, (list, tuple)):
            return int(item[int(batch_index)])
        return int(item)

    if torch.is_tensor(orig_shape):
        if orig_shape.ndim == 1 and int(orig_shape.numel()) == 3:
            vals = [int(x) for x in orig_shape.tolist()]
            return int(vals[0]), int(vals[1]), int(vals[2])
        if orig_shape.ndim >= 2 and int(orig_shape.shape[-1]) == 3:
            vals = [int(x) for x in orig_shape[int(batch_index)].tolist()]
            return int(vals[0]), int(vals[1]), int(vals[2])

    if isinstance(orig_shape, (list, tuple)):
        if len(orig_shape) == 3:
            vals = [_to_int(x) for x in orig_shape]
            return int(vals[0]), int(vals[1]), int(vals[2])
        if len(orig_shape) > int(batch_index):
            sample = orig_shape[int(batch_index)]
            if isinstance(sample, (list, tuple)) and len(sample) == 3:
                return int(sample[0]), int(sample[1]), int(sample[2])
            if torch.is_tensor(sample) and sample.ndim == 1 and int(sample.numel()) == 3:
                vals = [int(x) for x in sample.tolist()]
                return int(vals[0]), int(vals[1]), int(vals[2])
    return None


def _crop_tensor_to_orig_shape(x: torch.Tensor, orig_shape: tuple[int, int, int] | None) -> torch.Tensor:
    if orig_shape is None:
        return x
    orig_d, orig_h, orig_w = [int(v) for v in orig_shape]
    return x[..., :orig_d, :orig_h, :orig_w]


def _resolve_early_stop_value(
    *,
    metric_name: str,
    val_summary: dict[str, float | int],
    val_loss: float,
) -> float:
    metric_name = str(metric_name).strip()
    if metric_name == "val/dice_sw_mean":
        return float(val_summary["dice_mean"])
    if metric_name == "val/precision_sw_mean":
        return float(val_summary["precision_mean"])
    if metric_name == "val/recall_sw_mean":
        return float(val_summary["recall_mean"])
    if metric_name == "val/hd95_sw_mean":
        return float(val_summary["hd95_mean"])
    if metric_name == "val/rve_sw_mean":
        return float(val_summary["rve_mean"])
    if metric_name == "val/loss_sw":
        return float(val_loss)
    raise ValueError(f"Unsupported early_stop_metric: {metric_name}")


def _early_stop_improved(*, current: float, best: float | None, mode: str, min_delta: float) -> bool:
    if best is None or not math.isfinite(float(best)):
        return True
    if mode == "max":
        return float(current) > float(best) + float(min_delta)
    if mode == "min":
        return float(current) < float(best) - float(min_delta)
    raise ValueError(f"Unsupported early_stop_mode: {mode}")


def _overlay_masks_rgb(
    image2d: torch.Tensor,
    masks: torch.Tensor,
    *,
    alpha: float = 0.5,
    colors: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    image2d: (H,W) float in [0,1]
    masks: (C,H,W) float/bool in {0,1}
    returns: (3,H,W) float in [0,1]
    """
    if image2d.ndim != 2:
        raise ValueError(f"Expected image2d (H,W), got shape={tuple(image2d.shape)}")
    if masks.ndim != 3:
        raise ValueError(f"Expected masks (C,H,W), got shape={tuple(masks.shape)}")

    base = image2d.clamp(0, 1).unsqueeze(0).repeat(3, 1, 1)
    c = int(masks.shape[0])
    if colors is None:
        if c != 3:
            colors = torch.eye(3, dtype=base.dtype, device=base.device)[:c]
        else:
            colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=base.dtype, device=base.device)
    colors = colors.to(dtype=base.dtype, device=base.device)

    out = base
    for i in range(min(c, int(colors.shape[0]))):
        m = masks[i].float().clamp(0, 1).unsqueeze(0)
        col = colors[i].view(3, 1, 1)
        out = out * (1.0 - alpha * m) + col * (alpha * m)
    return out.clamp(0, 1)


def train_one_epoch(
    *,
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: torch.nn.Module,
    accelerator: Accelerator,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    global_step: int,
) -> tuple[float, float, int]:
    model.train()
    device = accelerator.device
    epoch_loss = 0.0
    epoch_grad_norm = 0.0

    for batch in train_loader:
        images, labels = _batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with accelerator.autocast():
            logits = model(images)
            loss = loss_function(logits, labels)

        accelerator.backward(loss)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))
        optimizer.step()
        scheduler.step()

        epoch_loss += float(loss.item())
        epoch_grad_norm += float(grad_norm.item())
        global_step += 1

    n = max(1, len(train_loader))
    return epoch_loss / n, epoch_grad_norm / n, global_step


@torch.no_grad()
def compute_patch_metrics(
    *,
    model: torch.nn.Module,
    data_loader: DataLoader,
    accelerator: Accelerator,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    model.eval()
    metric_acc = SegmentationMetricAccumulator(class_names=CLASSES, metric_names=["dice"], threshold=threshold)
    device = accelerator.device
    for batch in data_loader:
        images, labels = _batch_to_device(batch, device)
        with accelerator.autocast():
            logits = model(images)
        probs = torch.sigmoid(logits)
        preds = (probs > float(threshold)).float()
        metric_acc.update(preds, labels)
    return metric_acc.summary()


@torch.no_grad()
def compute_sw_metrics(
    *,
    model: torch.nn.Module,
    data_loader: DataLoader,
    accelerator: Accelerator,
    loss_function: torch.nn.Module,
    roi_size: tuple[int, int, int],
    sw_batch_size: int,
    sw_overlap: float,
    metric_names: Sequence[str] | None = None,
    threshold: float = 0.5,
    vis_case_day_to_slices: dict[str, list[int]] | None = None,
    logger: logging.Logger | None = None,
) -> tuple[dict[str, float | int], float, list[tuple[str, torch.Tensor]]]:
    model.eval()
    metric_acc = SegmentationMetricAccumulator(
        class_names=CLASSES,
        threshold=threshold,
        metric_names=metric_names or DEFAULT_METRIC_NAMES,
    )
    device = accelerator.device
    total_loss = 0.0
    vis_images: list[tuple[str, torch.Tensor]] = []

    total_batches = max(1, len(data_loader))
    for batch_index, batch in enumerate(data_loader, start=1):
        images, labels = _batch_to_device(batch, device)
        with accelerator.autocast():
            logits = sliding_window_inference(
                images,
                roi_size=roi_size,
                sw_batch_size=int(sw_batch_size),
                predictor=model,
                overlap=float(sw_overlap),
            )
        orig_shape = _extract_batch_orig_shape(batch, batch_index=0)
        logits = _crop_tensor_to_orig_shape(logits, orig_shape)
        labels = _crop_tensor_to_orig_shape(labels, orig_shape)
        with accelerator.autocast():
            loss = loss_function(logits, labels)
        total_loss += float(loss.item())

        probs = torch.sigmoid(logits)
        preds = (probs > float(threshold)).float()

        if logger is not None:
            logger.info(
                "Validation case %d/%d: inference done, computing metrics%s",
                batch_index,
                total_batches,
                " (including HD95)" if "hd95" in metric_acc.metric_names else "",
            )
        metric_acc.update(preds, labels)

        if vis_case_day_to_slices:
            case_day = _to_case_day(batch.get("case_day", ""))
            slice_positions = vis_case_day_to_slices.get(case_day)
            if slice_positions:
                img_vol = images[0, 0].detach()
                gt_vol = labels[0].detach()
                pred_vol = preds[0].detach()

                for slice_pos in slice_positions:
                    if slice_pos < 0 or slice_pos >= int(img_vol.shape[0]):
                        continue
                    img2d = img_vol[int(slice_pos)]
                    gt2d = gt_vol[:, int(slice_pos)]
                    pred2d = pred_vol[:, int(slice_pos)]

                    raw = img2d.clamp(0, 1).unsqueeze(0).repeat(3, 1, 1)
                    vis_images.append((f"val_vis/{case_day}/slice_{int(slice_pos):03d}/raw", raw))
                    vis_images.append(
                        (f"val_vis/{case_day}/slice_{int(slice_pos):03d}/gt", _overlay_masks_rgb(img2d, gt2d))
                    )
                    vis_images.append(
                        (f"val_vis/{case_day}/slice_{int(slice_pos):03d}/pred", _overlay_masks_rgb(img2d, pred2d))
                    )

        if logger is not None and (batch_index % 10 == 0 or batch_index == total_batches):
            logger.info("Validation progress: %d/%d cases", batch_index, total_batches)

    n = max(1, len(data_loader))
    return metric_acc.summary(), total_loss / n, vis_images


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/train_unet3d.py")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    cfg, cfg_module = _load_cfg(args.config)
    set_determinism(int(cfg.seed))

    out_dir = _resolve_output_dir(cfg)
    saved_config_path = _save_run_config(cfg, out_dir, config_path=args.config)
    logger = _setup_logger(out_dir / "train.log")
    writer = SummaryWriter(log_dir=str(out_dir / "tb"))

    mlflow_enabled = bool(cfg.use_mlflow)
    mlflow = None
    active_run = None
    if mlflow_enabled:
        try:
            dirty_status = _git_status_porcelain()
        except Exception as e:
            logger.error("Failed to query git status (%s).", str(e))
            logger.error("When use_mlflow=true, this script expects to run inside a git checkout.")
            raise SystemExit(2)

        if dirty_status.strip() and (not bool(cfg.allow_dirty_git)):
            logger.error("Refusing to run with dirty git worktree while use_mlflow=true and allow_dirty_git=false.")
            logger.error("Dirty files:\n%s", dirty_status)
            raise SystemExit(2)

        try:
            import mlflow as _mlflow  # type: ignore

            mlflow = _mlflow
        except Exception as e:
            raise RuntimeError("use_mlflow=true but mlflow import failed. Install mlflow or disable use_mlflow.") from e

        tracking_uri = _default_mlflow_tracking_uri(cfg)
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(_default_mlflow_experiment(cfg))

        resume_run_id = _load_mlflow_run_id_from_resume(cfg.resume_from)
        # Run name: deterministic enough to scan, but unique via timestamp in out_dir.
        run_name = f"{out_dir.name}"
        if resume_run_id:
            active_run = mlflow.start_run(run_id=resume_run_id)
        else:
            active_run = mlflow.start_run(run_name=run_name)
        run_id = active_run.info.run_id
        _write_mlflow_run_id(out_dir, run_id)

        # Tags: git + data + environment + linkage to output folder.
        mlflow.set_tags(_get_git_info())
        mlflow.set_tag("run_dir", str(out_dir))
        for k, v in _get_env_info().items():
            if v != "":
                mlflow.set_tag(k, v)

        # If dirty is allowed, capture a reproducible snapshot of local changes.
        if dirty_status.strip() and bool(cfg.allow_dirty_git):
            try:
                mlflow.set_tag("git.dirty_allowed", "true")
                (out_dir / "git_status_porcelain.txt").write_text(dirty_status + "\n", encoding="utf-8")
                mlflow.log_artifact(str(out_dir / "git_status_porcelain.txt"), artifact_path="git")
            except Exception:
                pass
            try:
                diff = _run_git(["diff"])
                (out_dir / "git_diff.patch").write_text(diff + "\n", encoding="utf-8")
                mlflow.log_artifact(str(out_dir / "git_diff.patch"), artifact_path="git")
            except Exception:
                pass
            try:
                diff_cached = _run_git(["diff", "--cached"])
                if diff_cached.strip():
                    (out_dir / "git_diff_cached.patch").write_text(diff_cached + "\n", encoding="utf-8")
                    mlflow.log_artifact(str(out_dir / "git_diff_cached.patch"), artifact_path="git")
            except Exception:
                pass

        # Data/config fingerprints.
        try:
            cfg_path = Path(args.config)
            if cfg_path.exists():
                mlflow.set_tag("config.path", str(cfg_path))
                mlflow.set_tag("config.sha256", _sha256_file(cfg_path))
        except Exception:
            pass
        for split_key, split_path in [("split.train", cfg.train_ids), ("split.val", cfg.val_ids)]:
            try:
                p = Path(split_path)
                if p.exists():
                    mlflow.set_tag(f"{split_key}.path", str(p))
                    mlflow.set_tag(f"{split_key}.sha256", _sha256_file(p))
            except Exception:
                pass
        if _cv_enabled(cfg):
            mlflow.set_tag("cv.enabled", "true")
            if str(cfg.cv_id).strip():
                mlflow.set_tag("cv.id", str(cfg.cv_id))
            if int(cfg.cv_fold) >= 0:
                mlflow.set_tag("cv.fold", str(int(cfg.cv_fold)))
            if int(cfg.cv_num_folds) > 0:
                mlflow.set_tag("cv.num_folds", str(int(cfg.cv_num_folds)))
            if str(cfg.cv_scheme).strip():
                mlflow.set_tag("cv.scheme", str(cfg.cv_scheme))
            if str(cfg.cv_group_by).strip():
                mlflow.set_tag("cv.group_by", str(cfg.cv_group_by))

        # Params: resolved config (flat key/value).
        try:
            mlflow.log_params({k: v for k, v in _cfg_to_serializable_dict(cfg).items()})
        except Exception:
            # Fallback: best-effort stringification (MLflow params are strings).
            mlflow.log_params({k: str(v) for k, v in _cfg_to_serializable_dict(cfg).items()})

        # Artifacts: configs/logs/splits/TB will be uploaded at end (and splits now).
        try:
            # Upload split files early to make the run self-contained.
            for p in [Path(cfg.train_ids), Path(cfg.val_ids)]:
                if p.exists():
                    mlflow.log_artifact(str(p), artifact_path="splits")
        except Exception:
            pass

    accelerator = Accelerator(mixed_precision=str(cfg.mixed_precision))
    logger.info("Device: %s | mixed_precision=%s", str(accelerator.device), str(cfg.mixed_precision))
    logger.info("Output: %s", str(out_dir))
    if _cv_enabled(cfg):
        logger.info(
            "CV: enabled=%s id=%s fold=%d/%d scheme=%s group_by=%s",
            str(bool(cfg.cv_enabled)).lower(),
            str(cfg.cv_id),
            int(cfg.cv_fold),
            int(cfg.cv_num_folds),
            str(cfg.cv_scheme),
            str(cfg.cv_group_by),
        )

    # For saving space: log static vis (raw/gt) only once per tag.
    logged_static_vis_tags: set[str] = set()

    # Running averages for utilization stats (when NVML is available).
    gpu_util_sum = 0.0
    gpu_util_n = 0
    gpu_mem_util_sum = 0.0
    gpu_mem_util_n = 0

    data_dir = os.path.join(cfg.data_root, "train")
    data_csv = os.path.join(cfg.data_root, "train.csv")

    logger.info("Building slice index...")
    case_day_slices = build_case_day_slices(data_dir)
    logger.info("Found case_day volumes: %d", len(case_day_slices))

    logger.info("Building RLE index...")
    rle_index = build_rle_index(data_csv)

    train_set = set(load_case_days(cfg.train_ids))
    val_set = set(load_case_days(cfg.val_ids))
    overlap = sorted(train_set & val_set)
    if overlap:
        raise ValueError(f"train_ids and val_ids overlap (n={len(overlap)}), e.g. {overlap[:5]}")

    train_files = [{"case_day": cd} for cd in sorted(train_set)]
    val_files = [{"case_day": cd} for cd in sorted(val_set)]
    logger.info("Train case_days: %d | Val case_days: %d", len(train_files), len(val_files))

    try:
        vis_samples = load_val_vis_samples(cfg.val_vis_samples)
        n_vis = len((vis_samples or {}).get("samples", []))
        logger.info("Validation visualization samples: %d", n_vis)
    except Exception as e:
        logger.info("val_vis_samples not loaded (%s): %s", cfg.val_vis_samples, str(e))
        vis_samples = None

    train_transforms, val_transforms = _build_transforms(
        cfg=cfg,
        cfg_module=cfg_module,
        case_day_slices=case_day_slices,
        rle_index=rle_index,
    )

    t_cache = time.time()
    logger.info("Caching train/val datasets... (workers=%d)", int(cfg.num_workers))
    train_ds = CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_rate=float(cfg.train_cache_rate),
        num_workers=int(cfg.num_workers),
        progress=False,
    )
    val_ds = CacheDataset(
        data=val_files,
        transform=val_transforms,
        cache_rate=float(cfg.val_cache_rate),
        num_workers=int(cfg.num_workers),
        progress=False,
    )
    logger.info("Caching time: %.2fs", time.time() - t_cache)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.train_batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        pin_memory=torch.cuda.is_available(),
        collate_fn=_pad_collate_keep_slice_idxs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.val_batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=torch.cuda.is_available(),
    )

    model = _build_model(cfg)
    loss_function = _build_loss(cfg=cfg, cfg_module=cfg_module)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    steps_per_epoch = max(1, len(train_loader))
    resolved_runtime_cfg = _write_resolved_runtime_config(out_dir, cfg, steps_per_epoch=int(steps_per_epoch))
    total_steps = int(resolved_runtime_cfg["total_steps"])
    val_interval_steps = int(resolved_runtime_cfg["val_interval_steps"])
    val_vis_every_steps = int(resolved_runtime_cfg["val_vis_every_steps"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(cfg.lr_max),
        epochs=int(cfg.epochs),
        steps_per_epoch=int(steps_per_epoch),
        pct_start=float(cfg.lr_pct_start),
        div_factor=float(cfg.lr_div_factor),
        final_div_factor=float(cfg.lr_final_div_factor),
    )

    start_epoch, global_step, best_metric, best_metric_epoch = _load_resume(
        cfg,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        logger=logger,
    )

    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)
    loss_function = loss_function.to(accelerator.device)

    logger.info("Loss: %s", loss_function.__class__.__name__)
    logger.info(
        "Schedule: steps_per_epoch=%d total_steps=%d val_interval_steps=%d val_vis_every_steps=%d",
        int(steps_per_epoch),
        int(total_steps),
        int(val_interval_steps),
        int(val_vis_every_steps),
    )
    writer.add_text(
        "run/equivalent_steps",
        (
            f"steps_per_epoch={int(steps_per_epoch)}\n"
            f"total_steps={int(total_steps)}\n"
            f"val_interval_steps={int(val_interval_steps)}\n"
            f"val_vis_every_steps={int(val_vis_every_steps)}"
        ),
        0,
    )
    writer.add_scalar("system/steps_per_epoch", float(steps_per_epoch), 0)
    writer.add_scalar("system/total_steps_planned", float(total_steps), 0)
    writer.add_scalar("system/val_interval_steps", float(val_interval_steps), 0)
    writer.add_scalar("system/val_vis_every_steps", float(val_vis_every_steps), 0)
    if mlflow_enabled and mlflow is not None:
        mlflow.log_params(
            {
                "steps_per_epoch": int(steps_per_epoch),
                "total_steps": int(total_steps),
                "val_interval_steps": int(val_interval_steps),
                "val_vis_every_steps": int(val_vis_every_steps),
            }
        )
        mlflow.set_tag("schedule.steps_per_epoch", str(int(steps_per_epoch)))
        mlflow.set_tag("schedule.total_steps", str(int(total_steps)))
        mlflow.set_tag("schedule.val_interval_steps", str(int(val_interval_steps)))
        mlflow.set_tag("schedule.val_vis_every_steps", str(int(val_vis_every_steps)))

    early_stop_enabled = bool(cfg.early_stop_enabled) and int(cfg.early_stop_patience) > 0
    early_stop_metric_name = str(cfg.early_stop_metric).strip()
    early_stop_mode = str(cfg.early_stop_mode).strip().lower()
    early_stop_min_epochs = max(0, int(cfg.early_stop_min_epochs))
    early_stop_patience = max(0, int(cfg.early_stop_patience))
    early_stop_min_delta = float(cfg.early_stop_min_delta)
    early_stop_best_value: float | None = None
    if early_stop_metric_name == "val/dice_sw_mean" and math.isfinite(float(best_metric)) and float(best_metric) >= 0.0:
        early_stop_best_value = float(best_metric)
    early_stop_bad_evals = 0
    early_stop_stop_epoch: int | None = None
    stop_reason = "max_epochs"

    training_start = time.time()
    for epoch in range(int(start_epoch), int(cfg.epochs) + 1):
        t0 = time.time()
        epoch_loss, epoch_grad_norm, global_step = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_function=loss_function,
            accelerator=accelerator,
            scheduler=scheduler,
            global_step=int(global_step),
        )
        train_time = time.time() - t0

        current_lr = float(optimizer.param_groups[0]["lr"])
        logger.info(
            "Epoch %d global_step=%d loss=%.4f lr=%.2e train_time=%.1fs",
            epoch,
            int(global_step),
            epoch_loss,
            current_lr,
            train_time,
        )
        writer.add_scalar("train/loss", float(epoch_loss), int(global_step))
        writer.add_scalar("train/lr", float(current_lr), int(global_step))
        writer.add_scalar("train/grad_norm", float(epoch_grad_norm), int(global_step))
        writer.add_scalar("system/train_time_sec", float(train_time), int(global_step))
        writer.add_scalar("system/epoch", float(epoch), int(global_step))
        if mlflow_enabled and mlflow is not None:
            mlflow.log_metric("train/loss", float(epoch_loss), step=int(global_step))
            mlflow.log_metric("train/lr", float(current_lr), step=int(global_step))
            mlflow.log_metric("train/grad_norm", float(epoch_grad_norm), step=int(global_step))
            mlflow.log_metric("system/train_time_sec", float(train_time), step=int(global_step))
            mlflow.log_metric("system/epoch", float(epoch), step=int(global_step))
        if torch.cuda.is_available():
            mem_alloc_gb = float(torch.cuda.memory_allocated()) / (1024 * 1024 * 1024)
            mem_res_gb = float(torch.cuda.memory_reserved()) / (1024 * 1024 * 1024)
            mem_max_alloc_gb = float(torch.cuda.max_memory_allocated()) / (1024 * 1024 * 1024)
            writer.add_scalar("system/gpu_mem_allocated_gb", mem_alloc_gb, int(global_step))
            writer.add_scalar("system/gpu_mem_reserved_gb", mem_res_gb, int(global_step))
            writer.add_scalar("system/gpu_mem_max_allocated_gb", mem_max_alloc_gb, int(global_step))
            if mlflow_enabled and mlflow is not None:
                mlflow.log_metric("system/gpu_mem_allocated_gb", mem_alloc_gb, step=int(global_step))
                mlflow.log_metric("system/gpu_mem_reserved_gb", mem_res_gb, step=int(global_step))
                mlflow.log_metric("system/gpu_mem_max_allocated_gb", mem_max_alloc_gb, step=int(global_step))
            gpu_stats = _read_gpu_stats()
            if gpu_stats:
                writer.add_scalar("system/gpu_utilization_pct", gpu_stats["gpu_utilization_pct"], int(global_step))
                writer.add_scalar("system/gpu_mem_utilization_pct", gpu_stats["gpu_mem_utilization_pct"], int(global_step))
                writer.add_scalar("system/gpu_mem_used_gb", gpu_stats["gpu_mem_used_gb"], int(global_step))

                gpu_util_sum += float(gpu_stats["gpu_utilization_pct"])
                gpu_util_n += 1
                gpu_mem_util_sum += float(gpu_stats["gpu_mem_utilization_pct"])
                gpu_mem_util_n += 1

                if mlflow_enabled and mlflow is not None:
                    mlflow.log_metric(
                        "system/gpu_utilization_pct",
                        float(gpu_stats["gpu_utilization_pct"]),
                        step=int(global_step),
                    )
                    mlflow.log_metric(
                        "system/gpu_mem_utilization_pct",
                        float(gpu_stats["gpu_mem_utilization_pct"]),
                        step=int(global_step),
                    )
                    mlflow.log_metric("system/gpu_mem_used_gb", float(gpu_stats["gpu_mem_used_gb"]), step=int(global_step))

        if epoch % int(cfg.val_interval) == 0:
            t_val = time.time()
            model.eval()
            run_train_patch_metrics = bool(cfg.train_patch_metrics_interval) and (
                epoch % int(cfg.train_patch_metrics_interval) == 0
            )
            run_hd95 = bool(cfg.val_hd95_interval) and (epoch % int(cfg.val_hd95_interval) == 0)
            val_metric_names = ["dice", "precision", "recall", "rve"]
            if run_hd95:
                val_metric_names.append("hd95")

            do_vis = bool(cfg.val_vis_every) and (epoch % int(cfg.val_vis_every) == 0) and bool(vis_samples)
            vis_case_day_to_slices: dict[str, list[int]] | None = None
            if do_vis:
                vis_case_day_to_slices = {}
                for s in (vis_samples or {}).get("samples", []):
                    cd = str(s.get("case_day") or "").strip()
                    if not cd:
                        continue
                    try:
                        sp = int(s.get("slice_pos", -1))
                    except Exception:
                        sp = -1
                    if sp < 0:
                        continue
                    vis_case_day_to_slices.setdefault(cd, []).append(sp)

            train_patch_summary = None
            if run_train_patch_metrics:
                train_patch_summary = compute_patch_metrics(
                    model=model,
                    data_loader=train_loader,
                    accelerator=accelerator,
                )
            val_sw_summary, val_sw_loss, vis_images = compute_sw_metrics(
                model=model,
                data_loader=val_loader,
                accelerator=accelerator,
                loss_function=loss_function,
                roi_size=cfg.patch_size,
                sw_batch_size=int(cfg.sw_batch_size),
                sw_overlap=float(cfg.sw_overlap),
                metric_names=val_metric_names,
                vis_case_day_to_slices=vis_case_day_to_slices,
                logger=logger,
            )
            val_time = time.time() - t_val

            train_patch_dice_mean = float(train_patch_summary["dice_mean"]) if train_patch_summary is not None else float("nan")
            val_sw_dice_mean = float(val_sw_summary["dice_mean"])
            val_sw_precision_mean = float(val_sw_summary["precision_mean"])
            val_sw_recall_mean = float(val_sw_summary["recall_mean"])
            val_sw_hd95_mean = float(val_sw_summary["hd95_mean"]) if "hd95_mean" in val_sw_summary else float("nan")
            val_sw_rve_mean = float(val_sw_summary["rve_mean"])
            early_stop_value = _resolve_early_stop_value(
                metric_name=early_stop_metric_name,
                val_summary=val_sw_summary,
                val_loss=float(val_sw_loss),
            )

            logger.info(
                "Epoch %d global_step=%d train_dice=%.4f val_dice=%.4f val_precision=%.4f val_recall=%.4f val_hd95=%.4f val_rve=%.4f val_loss=%.4f val_time=%.1fs",
                epoch,
                int(global_step),
                float(train_patch_dice_mean),
                float(val_sw_dice_mean),
                float(val_sw_precision_mean),
                float(val_sw_recall_mean),
                float(val_sw_hd95_mean),
                float(val_sw_rve_mean),
                float(val_sw_loss),
                float(val_time),
            )
            if train_patch_summary is not None:
                writer.add_scalar("train/dice_patch", float(train_patch_dice_mean), int(global_step))
            writer.add_scalar("val/dice_sw", float(val_sw_dice_mean), int(global_step))
            writer.add_scalar("val/precision_sw", float(val_sw_precision_mean), int(global_step))
            writer.add_scalar("val/recall_sw", float(val_sw_recall_mean), int(global_step))
            if "hd95_mean" in val_sw_summary:
                writer.add_scalar("val/hd95_sw", float(val_sw_hd95_mean), int(global_step))
            writer.add_scalar("val/rve_sw", float(val_sw_rve_mean), int(global_step))
            writer.add_scalar("val/loss_sw", float(val_sw_loss), int(global_step))
            writer.add_scalar("system/val_time_sec", float(val_time), int(global_step))
            writer.add_scalar("system/early_stop_metric", float(early_stop_value), int(global_step))
            if mlflow_enabled and mlflow is not None:
                if train_patch_summary is not None:
                    mlflow.log_metric("train/dice_patch_mean", float(train_patch_dice_mean), step=int(global_step))
                mlflow.log_metric("val/dice_sw_mean", float(val_sw_dice_mean), step=int(global_step))
                mlflow.log_metric("val/precision_sw_mean", float(val_sw_precision_mean), step=int(global_step))
                mlflow.log_metric("val/recall_sw_mean", float(val_sw_recall_mean), step=int(global_step))
                if "hd95_mean" in val_sw_summary:
                    mlflow.log_metric("val/hd95_sw_mean", float(val_sw_hd95_mean), step=int(global_step))
                mlflow.log_metric("val/rve_sw_mean", float(val_sw_rve_mean), step=int(global_step))
                mlflow.log_metric("val/loss_sw", float(val_sw_loss), step=int(global_step))
                mlflow.log_metric("system/val_time_sec", float(val_time), step=int(global_step))
                mlflow.log_metric("system/early_stop_metric", float(early_stop_value), step=int(global_step))

            for i, cls_name in enumerate(CLASSES):
                if train_patch_summary is not None:
                    writer.add_scalar(
                        f"train/dice_patch_{cls_name}",
                        float(train_patch_summary[f"dice_{cls_name}"]),
                        int(global_step),
                    )
                writer.add_scalar(f"val/dice_sw_{cls_name}", float(val_sw_summary[f"dice_{cls_name}"]), int(global_step))
                writer.add_scalar(
                    f"val/precision_sw_{cls_name}",
                    float(val_sw_summary[f"precision_{cls_name}"]),
                    int(global_step),
                )
                writer.add_scalar(
                    f"val/recall_sw_{cls_name}",
                    float(val_sw_summary[f"recall_{cls_name}"]),
                    int(global_step),
                )
                if f"hd95_{cls_name}" in val_sw_summary:
                    writer.add_scalar(f"val/hd95_sw_{cls_name}", float(val_sw_summary[f"hd95_{cls_name}"]), int(global_step))
                writer.add_scalar(f"val/rve_sw_{cls_name}", float(val_sw_summary[f"rve_{cls_name}"]), int(global_step))
                if mlflow_enabled and mlflow is not None:
                    if train_patch_summary is not None:
                        mlflow.log_metric(
                            f"train/dice_patch_{cls_name}",
                            float(train_patch_summary[f"dice_{cls_name}"]),
                            step=int(global_step),
                        )
                    mlflow.log_metric(
                        f"val/dice_sw_{cls_name}",
                        float(val_sw_summary[f"dice_{cls_name}"]),
                        step=int(global_step),
                    )
                    mlflow.log_metric(
                        f"val/precision_sw_{cls_name}",
                        float(val_sw_summary[f"precision_{cls_name}"]),
                        step=int(global_step),
                    )
                    mlflow.log_metric(
                        f"val/recall_sw_{cls_name}",
                        float(val_sw_summary[f"recall_{cls_name}"]),
                        step=int(global_step),
                    )
                    if f"hd95_{cls_name}" in val_sw_summary:
                        mlflow.log_metric(
                            f"val/hd95_sw_{cls_name}",
                            float(val_sw_summary[f"hd95_{cls_name}"]),
                            step=int(global_step),
                        )
                    mlflow.log_metric(
                        f"val/rve_sw_{cls_name}",
                        float(val_sw_summary[f"rve_{cls_name}"]),
                        step=int(global_step),
                    )

            if do_vis and vis_images:
                for tag, img in vis_images:
                    # Save raw/gt once; pred can evolve over time.
                    if tag.endswith("/raw") or tag.endswith("/gt"):
                        if tag in logged_static_vis_tags:
                            continue
                        logged_static_vis_tags.add(tag)
                    writer.add_image(tag, img.detach().cpu(), int(global_step), dataformats="CHW")

            if float(val_sw_dice_mean) > float(best_metric):
                best_metric = float(val_sw_dice_mean)
                best_metric_epoch = int(epoch)
                unwrapped = accelerator.unwrap_model(model)
                torch.save(
                    {
                        "epoch": int(epoch),
                        "global_step": int(global_step),
                        "model_state_dict": unwrapped.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_metric": float(best_metric),
                        "best_metric_epoch": int(best_metric_epoch),
                        "metric_summary": {k: float(v) if isinstance(v, float) else int(v) for k, v in val_sw_summary.items()},
                    },
                    str(out_dir / "best.pt"),
                )
                logger.info("Saved new best model -> %s", str(out_dir / "best.pt"))

            if early_stop_enabled:
                improved = _early_stop_improved(
                    current=float(early_stop_value),
                    best=early_stop_best_value,
                    mode=early_stop_mode,
                    min_delta=float(early_stop_min_delta),
                )
                if improved:
                    early_stop_best_value = float(early_stop_value)
                    early_stop_bad_evals = 0
                elif int(epoch) >= int(early_stop_min_epochs):
                    early_stop_bad_evals += 1
                logger.info(
                    "Early stop monitor: metric=%s mode=%s value=%.6f best=%.6f bad_evals=%d/%d min_epochs=%d",
                    early_stop_metric_name,
                    early_stop_mode,
                    float(early_stop_value),
                    float(early_stop_best_value) if early_stop_best_value is not None else float("nan"),
                    int(early_stop_bad_evals),
                    int(early_stop_patience),
                    int(early_stop_min_epochs),
                )
                if int(epoch) >= int(early_stop_min_epochs) and int(early_stop_bad_evals) >= int(early_stop_patience):
                    early_stop_stop_epoch = int(epoch)
                    stop_reason = "early_stop"

        unwrapped = accelerator.unwrap_model(model)
        torch.save(
            {
                "epoch": int(epoch + 1),
                "global_step": int(global_step),
                "model_state_dict": unwrapped.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_metric": float(best_metric),
                "best_metric_epoch": int(best_metric_epoch),
            },
            str(out_dir / "last.pt"),
        )

        if early_stop_stop_epoch is not None:
            logger.info("Early stopping triggered at epoch=%d", int(early_stop_stop_epoch))
            break

    total = time.time() - training_start
    logger.info("Training done. best_metric=%.4f at epoch=%d", float(best_metric), int(best_metric_epoch))
    logger.info("Total time: %.1fs (%.2fh)", float(total), float(total) / 3600.0)
    total_hr = float(total) / 3600.0
    writer.add_scalar("system/total_training_time_hr", float(total_hr), int(global_step))
    if gpu_util_n:
        writer.add_scalar(
            "system/avg_gpu_utilization_pct",
            float(gpu_util_sum) / float(gpu_util_n),
            int(global_step),
        )
    if gpu_mem_util_n:
        writer.add_scalar(
            "system/avg_gpu_mem_utilization_pct",
            float(gpu_mem_util_sum) / float(gpu_mem_util_n),
            int(global_step),
        )
    if mlflow_enabled and mlflow is not None:
        mlflow.log_metric("system/total_training_time_hr", float(total_hr), step=int(global_step))
        mlflow.log_metric("summary/best_metric", float(best_metric), step=int(global_step))
        mlflow.set_tag("summary/best_metric_epoch", str(int(best_metric_epoch)))
        mlflow.set_tag("summary/last_global_step", str(int(global_step)))
        mlflow.set_tag("summary/stop_reason", str(stop_reason))
        mlflow.set_tag("summary/stopped_early", "true" if stop_reason == "early_stop" else "false")
        if gpu_util_n:
            mlflow.log_metric("summary/avg_gpu_utilization_pct", float(gpu_util_sum) / float(gpu_util_n), step=int(global_step))
        if gpu_mem_util_n:
            mlflow.log_metric(
                "summary/avg_gpu_mem_utilization_pct",
                float(gpu_mem_util_sum) / float(gpu_mem_util_n),
                step=int(global_step),
            )
    writer.close()

    if mlflow_enabled and mlflow is not None:
        # Upload core artifacts for reproducibility and analysis.
        try:
            mlflow.log_artifact(str(saved_config_path), artifact_path="run")
        except Exception:
            pass
        try:
            mlflow.log_artifact(str(out_dir / "config.resolved.json"), artifact_path="run")
        except Exception:
            pass
        try:
            mlflow.log_artifact(str(out_dir / "train.log"), artifact_path="run")
        except Exception:
            pass
        for ckpt_name in ["best.pt", "last.pt"]:
            try:
                p = out_dir / ckpt_name
                if p.exists():
                    mlflow.log_artifact(str(p), artifact_path="checkpoints")
            except Exception:
                pass
        try:
            tb_dir = out_dir / "tb"
            if tb_dir.exists():
                mlflow.log_artifacts(str(tb_dir), artifact_path="tb")
        except Exception:
            pass
        try:
            mlflow.end_run()
        except Exception:
            pass


if __name__ == "__main__":
    main()
