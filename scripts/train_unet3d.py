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
import json
import logging
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
import yaml
from accelerate import Accelerator
from monai.data import CacheDataset, DataLoader
from monai.data import pad_list_data_collate
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.layers import Norm
from monai.networks.nets import Unet
from monai.transforms import Compose, EnsureChannelFirstd, EnsureTyped, RandSpatialCropd, ScaleIntensityd, SpatialPadd
from monai.utils import set_determinism
from torch.utils.tensorboard import SummaryWriter

from src.constants import CLASSES
from src.data_utils import build_case_day_slices, build_rle_index, load_case_days, load_val_vis_samples
from src.datasets import LoadCaseDayVolumed

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

    # OneCycleLR
    lr_max: float = 5e-4
    lr_pct_start: float = 0.2
    lr_div_factor: float = 25.0
    lr_final_div_factor: float = 1000.0

    # training
    epochs: int = 150
    val_interval: int = 3
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


def _load_cfg(path: str) -> TrainCfg:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError("Config YAML must be a mapping at the top level.")

    known = set(TrainCfg().__dict__.keys())
    unknown = sorted([k for k in raw.keys() if k not in known])
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")

    merged = {**TrainCfg().__dict__, **raw}
    return TrainCfg(**merged)


def _resolve_output_dir(cfg: TrainCfg) -> Path:
    if cfg.output_dir:
        return Path(cfg.output_dir)
    ts = time.strftime("%Y%m%d-%H%M%S")
    return Path("outputs") / f"{ts}_{cfg.model_name}"


def _save_run_config(cfg: TrainCfg, out_dir: Path, *, config_path: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(Path(config_path).read_text(encoding="utf-8"), encoding="utf-8")
    (out_dir / "config.resolved.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")


def _build_transforms(*, cfg: TrainCfg, case_day_slices, rle_index):
    common = [
        LoadCaseDayVolumed(
            keys=["case_day"],
            case_day_slices=case_day_slices,
            rle_index=rle_index,
        ),
        EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        SpatialPadd(keys=["image", "label"], spatial_size=cfg.patch_size, mode="constant"),
    ]

    train_transforms = Compose(
        [
            *common,
            RandSpatialCropd(keys=["image", "label"], roi_size=cfg.patch_size, random_size=False),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
    val_transforms = Compose([*common, EnsureTyped(keys=["image", "label"])])
    return train_transforms, val_transforms


def _pad_collate_keep_slice_idxs(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad-collate tensors but keep variable-length slice_idxs as a list."""
    slice_idxs = [item.get("slice_idxs") for item in batch]
    batch_no_idxs: list[dict[str, Any]] = []
    for item in batch:
        item = dict(item)
        item.pop("slice_idxs", None)
        batch_no_idxs.append(item)
    collated = pad_list_data_collate(batch_no_idxs)
    collated["slice_idxs"] = slice_idxs
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


def _load_resume(
    cfg: TrainCfg,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    logger: logging.Logger,
) -> tuple[int, float, int]:
    if not cfg.resume_from:
        return 1, -1.0, -1

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
    best_metric = float(ckpt.get("best_metric", -1.0))
    best_metric_epoch = int(ckpt.get("best_metric_epoch", -1))
    logger.info("Resumed from %s (epoch=%d, best=%.4f@%d)", str(p), start_epoch, best_metric, best_metric_epoch)
    return start_epoch, best_metric, best_metric_epoch


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
) -> tuple[float, float]:
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

    n = max(1, len(train_loader))
    return epoch_loss / n, epoch_grad_norm / n


@torch.no_grad()
def compute_patch_dice(
    *,
    model: torch.nn.Module,
    data_loader: DataLoader,
    accelerator: Accelerator,
    dice_metric: DiceMetric,
    threshold: float = 0.5,
) -> tuple[torch.Tensor, float]:
    model.eval()
    device = accelerator.device
    for batch in data_loader:
        images, labels = _batch_to_device(batch, device)
        with accelerator.autocast():
            logits = model(images)
        probs = torch.sigmoid(logits)
        preds = (probs > float(threshold)).float()
        dice_metric(y_pred=preds, y=labels)
    dice_per_class = dice_metric.aggregate()
    return dice_per_class, float(dice_per_class.mean().item())


@torch.no_grad()
def compute_sw_dice(
    *,
    model: torch.nn.Module,
    data_loader: DataLoader,
    accelerator: Accelerator,
    dice_metric: DiceMetric,
    loss_function: torch.nn.Module,
    roi_size: tuple[int, int, int],
    sw_batch_size: int,
    sw_overlap: float,
    threshold: float = 0.5,
    vis_case_day_to_slices: dict[str, list[int]] | None = None,
) -> tuple[torch.Tensor, float, float, list[tuple[str, torch.Tensor]]]:
    model.eval()
    device = accelerator.device
    total_loss = 0.0
    vis_images: list[tuple[str, torch.Tensor]] = []

    for batch in data_loader:
        images, labels = _batch_to_device(batch, device)
        with accelerator.autocast():
            logits = sliding_window_inference(
                images,
                roi_size=roi_size,
                sw_batch_size=int(sw_batch_size),
                predictor=model,
                overlap=float(sw_overlap),
            )
            loss = loss_function(logits, labels)
        total_loss += float(loss.item())

        probs = torch.sigmoid(logits)
        preds = (probs > float(threshold)).float()
        dice_metric(y_pred=preds, y=labels)

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

    dice_per_class = dice_metric.aggregate()
    dice_mean = float(dice_per_class.mean().item())
    n = max(1, len(data_loader))
    return dice_per_class, dice_mean, total_loss / n, vis_images


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/train_unet3d.yaml")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    cfg = _load_cfg(args.config)
    set_determinism(int(cfg.seed))

    out_dir = _resolve_output_dir(cfg)
    _save_run_config(cfg, out_dir, config_path=args.config)
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

        # Params: resolved config (flat key/value).
        try:
            mlflow.log_params({k: v for k, v in asdict(cfg).items()})
        except Exception:
            # Fallback: best-effort stringification (MLflow params are strings).
            mlflow.log_params({k: str(v) for k, v in asdict(cfg).items()})

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

    train_transforms, val_transforms = _build_transforms(cfg=cfg, case_day_slices=case_day_slices, rle_index=rle_index)

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
    loss_function = DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(cfg.lr_max),
        epochs=int(cfg.epochs),
        steps_per_epoch=int(steps_per_epoch),
        pct_start=float(cfg.lr_pct_start),
        div_factor=float(cfg.lr_div_factor),
        final_div_factor=float(cfg.lr_final_div_factor),
    )

    start_epoch, best_metric, best_metric_epoch = _load_resume(
        cfg,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        logger=logger,
    )

    dice_metric = DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=False)

    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)
    loss_function = loss_function.to(accelerator.device)

    training_start = time.time()
    for epoch in range(int(start_epoch), int(cfg.epochs) + 1):
        t0 = time.time()
        epoch_loss, epoch_grad_norm = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_function=loss_function,
            accelerator=accelerator,
            scheduler=scheduler,
        )
        train_time = time.time() - t0

        current_lr = float(optimizer.param_groups[0]["lr"])
        logger.info("Epoch %d loss=%.4f lr=%.2e train_time=%.1fs", epoch, epoch_loss, current_lr, train_time)
        writer.add_scalar("train/loss", float(epoch_loss), epoch)
        writer.add_scalar("train/lr", float(current_lr), epoch)
        writer.add_scalar("train/grad_norm", float(epoch_grad_norm), epoch)
        writer.add_scalar("system/train_time_sec", float(train_time), epoch)
        if mlflow_enabled and mlflow is not None:
            mlflow.log_metric("train/loss", float(epoch_loss), step=int(epoch))
            mlflow.log_metric("train/lr", float(current_lr), step=int(epoch))
            mlflow.log_metric("train/grad_norm", float(epoch_grad_norm), step=int(epoch))
            mlflow.log_metric("system/train_time_sec", float(train_time), step=int(epoch))
        if torch.cuda.is_available():
            mem_alloc_gb = float(torch.cuda.memory_allocated()) / (1024 * 1024 * 1024)
            mem_res_gb = float(torch.cuda.memory_reserved()) / (1024 * 1024 * 1024)
            mem_max_alloc_gb = float(torch.cuda.max_memory_allocated()) / (1024 * 1024 * 1024)
            writer.add_scalar("system/gpu_mem_allocated_gb", mem_alloc_gb, epoch)
            writer.add_scalar("system/gpu_mem_reserved_gb", mem_res_gb, epoch)
            writer.add_scalar("system/gpu_mem_max_allocated_gb", mem_max_alloc_gb, epoch)
            if mlflow_enabled and mlflow is not None:
                mlflow.log_metric("system/gpu_mem_allocated_gb", mem_alloc_gb, step=int(epoch))
                mlflow.log_metric("system/gpu_mem_reserved_gb", mem_res_gb, step=int(epoch))
                mlflow.log_metric("system/gpu_mem_max_allocated_gb", mem_max_alloc_gb, step=int(epoch))
            gpu_stats = _read_gpu_stats()
            if gpu_stats:
                writer.add_scalar("system/gpu_utilization_pct", gpu_stats["gpu_utilization_pct"], epoch)
                writer.add_scalar("system/gpu_mem_utilization_pct", gpu_stats["gpu_mem_utilization_pct"], epoch)
                writer.add_scalar("system/gpu_mem_used_gb", gpu_stats["gpu_mem_used_gb"], epoch)

                gpu_util_sum += float(gpu_stats["gpu_utilization_pct"])
                gpu_util_n += 1
                gpu_mem_util_sum += float(gpu_stats["gpu_mem_utilization_pct"])
                gpu_mem_util_n += 1

                if mlflow_enabled and mlflow is not None:
                    mlflow.log_metric("system/gpu_utilization_pct", float(gpu_stats["gpu_utilization_pct"]), step=int(epoch))
                    mlflow.log_metric(
                        "system/gpu_mem_utilization_pct",
                        float(gpu_stats["gpu_mem_utilization_pct"]),
                        step=int(epoch),
                    )
                    mlflow.log_metric("system/gpu_mem_used_gb", float(gpu_stats["gpu_mem_used_gb"]), step=int(epoch))

        if epoch % int(cfg.val_interval) == 0:
            t_val = time.time()
            model.eval()

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

            dice_metric.reset()
            train_patch_dice_per_class, train_patch_dice_mean = compute_patch_dice(
                model=model,
                data_loader=train_loader,
                accelerator=accelerator,
                dice_metric=dice_metric,
            )
            dice_metric.reset()
            val_sw_dice_per_class, val_sw_dice_mean, val_sw_loss, vis_images = compute_sw_dice(
                model=model,
                data_loader=val_loader,
                accelerator=accelerator,
                dice_metric=dice_metric,
                loss_function=loss_function,
                roi_size=cfg.patch_size,
                sw_batch_size=int(cfg.sw_batch_size),
                sw_overlap=float(cfg.sw_overlap),
                vis_case_day_to_slices=vis_case_day_to_slices,
            )
            dice_metric.reset()
            val_time = time.time() - t_val

            logger.info(
                "Epoch %d train_dice=%.4f val_sw_dice=%.4f val_sw_loss=%.4f val_time=%.1fs",
                epoch,
                float(train_patch_dice_mean),
                float(val_sw_dice_mean),
                float(val_sw_loss),
                float(val_time),
            )
            writer.add_scalar("train/dice_patch", float(train_patch_dice_mean), epoch)
            writer.add_scalar("val/dice_sw", float(val_sw_dice_mean), epoch)
            writer.add_scalar("val/loss_sw", float(val_sw_loss), epoch)
            writer.add_scalar("system/val_time_sec", float(val_time), epoch)
            if mlflow_enabled and mlflow is not None:
                mlflow.log_metric("train/dice_patch_mean", float(train_patch_dice_mean), step=int(epoch))
                mlflow.log_metric("val/dice_sw_mean", float(val_sw_dice_mean), step=int(epoch))
                mlflow.log_metric("val/loss_sw", float(val_sw_loss), step=int(epoch))
                mlflow.log_metric("system/val_time_sec", float(val_time), step=int(epoch))

            for i, cls_name in enumerate(CLASSES):
                writer.add_scalar(f"train/dice_patch_{cls_name}", float(train_patch_dice_per_class[i].item()), epoch)
                writer.add_scalar(f"val/dice_sw_{cls_name}", float(val_sw_dice_per_class[i].item()), epoch)
                if mlflow_enabled and mlflow is not None:
                    mlflow.log_metric(
                        f"train/dice_patch_{cls_name}",
                        float(train_patch_dice_per_class[i].item()),
                        step=int(epoch),
                    )
                    mlflow.log_metric(
                        f"val/dice_sw_{cls_name}",
                        float(val_sw_dice_per_class[i].item()),
                        step=int(epoch),
                    )

            if do_vis and vis_images:
                for tag, img in vis_images:
                    # Save raw/gt once; pred can evolve over time.
                    if tag.endswith("/raw") or tag.endswith("/gt"):
                        if tag in logged_static_vis_tags:
                            continue
                        logged_static_vis_tags.add(tag)
                    writer.add_image(tag, img.detach().cpu(), epoch, dataformats="CHW")

            if float(val_sw_dice_mean) > float(best_metric):
                best_metric = float(val_sw_dice_mean)
                best_metric_epoch = int(epoch)
                unwrapped = accelerator.unwrap_model(model)
                torch.save(
                    {
                        "epoch": int(epoch),
                        "model_state_dict": unwrapped.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_metric": float(best_metric),
                        "best_metric_epoch": int(best_metric_epoch),
                        "dice_per_class": val_sw_dice_per_class.detach().cpu().numpy(),
                    },
                    str(out_dir / "best.pt"),
                )
                logger.info("Saved new best model -> %s", str(out_dir / "best.pt"))

        unwrapped = accelerator.unwrap_model(model)
        torch.save(
            {
                "epoch": int(epoch + 1),
                "model_state_dict": unwrapped.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_metric": float(best_metric),
                "best_metric_epoch": int(best_metric_epoch),
            },
            str(out_dir / "last.pt"),
        )

    total = time.time() - training_start
    logger.info("Training done. best_metric=%.4f at epoch=%d", float(best_metric), int(best_metric_epoch))
    logger.info("Total time: %.1fs (%.2fh)", float(total), float(total) / 3600.0)
    total_hr = float(total) / 3600.0
    writer.add_scalar("system/total_training_time_hr", float(total_hr), int(cfg.epochs))
    if gpu_util_n:
        writer.add_scalar(
            "system/avg_gpu_utilization_pct",
            float(gpu_util_sum) / float(gpu_util_n),
            int(cfg.epochs),
        )
    if gpu_mem_util_n:
        writer.add_scalar(
            "system/avg_gpu_mem_utilization_pct",
            float(gpu_mem_util_sum) / float(gpu_mem_util_n),
            int(cfg.epochs),
        )
    if mlflow_enabled and mlflow is not None:
        mlflow.log_metric("system/total_training_time_hr", float(total_hr), step=int(cfg.epochs))
        mlflow.log_metric("summary/best_metric", float(best_metric), step=int(cfg.epochs))
        mlflow.set_tag("summary/best_metric_epoch", str(int(best_metric_epoch)))
        if gpu_util_n:
            mlflow.log_metric("summary/avg_gpu_utilization_pct", float(gpu_util_sum) / float(gpu_util_n), step=int(cfg.epochs))
        if gpu_mem_util_n:
            mlflow.log_metric(
                "summary/avg_gpu_mem_utilization_pct",
                float(gpu_mem_util_sum) / float(gpu_mem_util_n),
                step=int(cfg.epochs),
            )
    writer.close()

    if mlflow_enabled and mlflow is not None:
        # Upload core artifacts for reproducibility and analysis.
        try:
            mlflow.log_artifact(str(out_dir / "config.yaml"), artifact_path="run")
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
