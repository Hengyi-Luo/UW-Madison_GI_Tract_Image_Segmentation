#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
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


@dataclass(frozen=True)
class TrainCfg:
    # data
    data_root: str = "./inputs"
    train_ids: str = "./inputs/splits/train_case_days.csv"
    val_ids: str = "./inputs/splits/val_case_days.csv"
    num_workers: int = 6
    scan_read_mode: str = "unchanged"  # unchanged | legacy_uint8
    seed: int = 42

    # checkpointing
    output_dir: str = ""
    resume_from: str = ""

    # model
    model_name: str = "Unet3D"
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
    val_vis_samples: str = "./inputs/splits/val_vis_samples.json"

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
            scan_read_mode=str(cfg.scan_read_mode),
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
) -> tuple[torch.Tensor, float, float]:
    model.eval()
    device = accelerator.device
    total_loss = 0.0

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

    dice_per_class = dice_metric.aggregate()
    dice_mean = float(dice_per_class.mean().item())
    n = max(1, len(data_loader))
    return dice_per_class, dice_mean, total_loss / n


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

    accelerator = Accelerator(mixed_precision=str(cfg.mixed_precision))
    logger.info("Device: %s | mixed_precision=%s", str(accelerator.device), str(cfg.mixed_precision))
    logger.info("Output: %s", str(out_dir))

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
        collate_fn=pad_list_data_collate,
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

        if epoch % int(cfg.val_interval) == 0:
            t_val = time.time()
            model.eval()

            dice_metric.reset()
            train_patch_dice_per_class, train_patch_dice_mean = compute_patch_dice(
                model=model,
                data_loader=train_loader,
                accelerator=accelerator,
                dice_metric=dice_metric,
            )
            dice_metric.reset()
            val_sw_dice_per_class, val_sw_dice_mean, val_sw_loss = compute_sw_dice(
                model=model,
                data_loader=val_loader,
                accelerator=accelerator,
                dice_metric=dice_metric,
                loss_function=loss_function,
                roi_size=cfg.patch_size,
                sw_batch_size=int(cfg.sw_batch_size),
                sw_overlap=float(cfg.sw_overlap),
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

            for i, cls_name in enumerate(CLASSES):
                writer.add_scalar(f"train/dice_patch_{cls_name}", float(train_patch_dice_per_class[i].item()), epoch)
                writer.add_scalar(f"val/dice_sw_{cls_name}", float(val_sw_dice_per_class[i].item()), epoch)

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
                        "scan_read_mode": str(cfg.scan_read_mode),
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
                "scan_read_mode": str(cfg.scan_read_mode),
            },
            str(out_dir / "last.pt"),
        )

    total = time.time() - training_start
    logger.info("Training done. best_metric=%.4f at epoch=%d", float(best_metric), int(best_metric_epoch))
    logger.info("Total time: %.1fs (%.2fh)", float(total), float(total) / 3600.0)
    writer.add_scalar("system/total_training_time_sec", float(total), int(cfg.epochs))
    writer.close()


if __name__ == "__main__":
    main()
