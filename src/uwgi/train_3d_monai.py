from __future__ import annotations

import os
import time
from dataclasses import dataclass

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.utils import set_determinism

from .checkpointing import (
    apply_cfg_from_checkpoint,
    checkpoint_paths,
    load_init_weights,
    load_resume_checkpoint,
    move_optimizer_state_to_device,
)
from .constants import CLASSES
from .data_utils import build_case_day_slices, build_rle_index
from .datasets import UWGI3DPatchDataset
from .engine import train_one_epoch, validate_dice
from .experiment import git_info, resolve_run_naming
from .models import build_model
from .splits import split_case_days


@dataclass
class TrainCfg:
    # Data
    data_root: str = "./input/uw-madison-gi-tract-image-segmentation"
    seed: int = 42
    val_ratio: float = 0.2
    num_workers: int = 4
    cache_in_ram: bool = False

    # Patches
    patch_d: int = 96
    patch_h: int = 224
    patch_w: int = 224
    samples_per_volume: int = 12

    # Model
    model: str = "swin_unetr"  # unet | swin_unetr
    feature_size: int = 48
    use_checkpoint: bool = False
    init_from: str = ""  # optional pretrained init (shape-matched)

    # Optimization
    epochs: int | None = 30
    batch_size: int = 1
    lr: float = 1e-4

    # Output / resume
    out: str = "./checkpoints/best.pt"
    run_dir: str = ""
    resume_from: str = ""  # path to .../last.pt
    save_last_every: int = 1

    # Runtime
    mixed_precision: str = "no"  # "no", "fp16", "bf16"
    verbose_shape_fix: bool = False

    # Logging
    use_tensorboard: bool = False
    tb_dir: str = "./outputs/train_run/tb"
    log_steps: int = 50
    use_mlflow: bool = True
    mlflow_tracking_uri: str = ""
    mlflow_experiment: str = "uwgi"
    mlflow_run_name: str = ""
    mlflow_run_id: str = ""
    debug: bool = False


def train(cfg: TrainCfg) -> None:
    set_determinism(seed=cfg.seed)
    ckpt = load_resume_checkpoint(cfg.resume_from)

    if ckpt is not None and cfg.resume_from:
        apply_cfg_from_checkpoint(cfg, ckpt)
        resume_dir = os.path.dirname(cfg.resume_from)
        if (not cfg.run_dir) or ("{" in cfg.run_dir and "}" in cfg.run_dir):
            cfg.run_dir = resume_dir
        if not cfg.mlflow_run_id:
            cfg.mlflow_run_id = ckpt.get("mlflow_run_id", "") or ckpt.get("mlflow", {}).get("run_id", "")

    resolve_run_naming(cfg)
    if cfg.run_dir:
        cfg.out = os.path.join(cfg.run_dir, "best.pt")
        if cfg.use_tensorboard:
            cfg.tb_dir = os.path.join(cfg.run_dir, "tb")
    os.makedirs(os.path.dirname(cfg.out), exist_ok=True)

    train_dir = os.path.join(cfg.data_root, "train")
    train_csv = os.path.join(cfg.data_root, "train.csv")

    print("Building slice index...")
    case_day_slices = build_case_day_slices(train_dir)
    all_case_days = sorted(case_day_slices.keys())
    print(f"Found case_day volumes: {len(all_case_days)}")

    print("Building RLE index...")
    rle_index = build_rle_index(train_csv)

    train_days, val_days = split_case_days(all_case_days, val_ratio=cfg.val_ratio, seed=cfg.seed)
    print(f"Train volumes: {len(train_days)} | Val volumes: {len(val_days)}")

    patch_size = (int(cfg.patch_d), int(cfg.patch_h), int(cfg.patch_w))
    train_ds = UWGI3DPatchDataset(
        case_days=train_days,
        case_day_slices=case_day_slices,
        rle_index=rle_index,
        patch_size=patch_size,
        samples_per_volume=int(cfg.samples_per_volume),
        cache_in_ram=bool(cfg.cache_in_ram),
        seed=int(cfg.seed),
        is_train=True,
        verbose_shape_fix=bool(cfg.verbose_shape_fix),
    )
    val_ds = UWGI3DPatchDataset(
        case_days=val_days,
        case_day_slices=case_day_slices,
        rle_index=rle_index,
        patch_size=patch_size,
        samples_per_volume=max(1, int(cfg.samples_per_volume) // 2),
        cache_in_ram=bool(cfg.cache_in_ram),
        seed=int(cfg.seed) + 999,
        is_train=False,
        verbose_shape_fix=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=True,
    )

    accelerator = Accelerator(mixed_precision=None if cfg.mixed_precision == "no" else cfg.mixed_precision)
    device = accelerator.device

    model = build_model(
        model_name=cfg.model,
        patch_size=patch_size,
        in_channels=1,
        out_channels=len(CLASSES),
        feature_size=int(cfg.feature_size),
        use_checkpoint=bool(cfg.use_checkpoint),
    ).to(device)

    loss_fn = DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=1e-4)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    start_epoch = 1
    best_dice = -1.0
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_dice = float(ckpt.get("best_dice", -1.0))
    else:
        loaded = load_init_weights(model, cfg.init_from)
        if loaded:
            print(f"Loaded init weights tensors: {loaded}")

    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

    if ckpt is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        move_optimizer_state_to_device(optimizer, accelerator.device)

    mlflow = None
    mlflow_active = False
    mlflow_run_id = ""
    if cfg.use_mlflow and accelerator.is_main_process:
        try:
            import mlflow as _mlflow
        except Exception:
            raise RuntimeError("MLflow not installed. Please: pip install mlflow")
        mlflow = _mlflow
        if cfg.mlflow_tracking_uri:
            mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
        if cfg.mlflow_experiment:
            mlflow.set_experiment(cfg.mlflow_experiment)

        resume_same_run = bool(cfg.mlflow_run_id)
        if resume_same_run:
            mlflow.start_run(run_id=cfg.mlflow_run_id)
        else:
            mlflow.start_run(run_name=cfg.mlflow_run_name)
        mlflow_run_id = mlflow.active_run().info.run_id

        git = git_info(os.getcwd())
        if git.get("sha"):
            mlflow.set_tag("git_sha", git["sha"])
        if git.get("branch"):
            mlflow.set_tag("git_branch", git["branch"])
        if "dirty" in git:
            mlflow.set_tag("git_dirty", str(bool(git["dirty"])))

        if not resume_same_run:
            for k, v in cfg.__dict__.items():
                if k in {"resume_from", "mlflow_run_id"}:
                    continue
                mlflow.log_param(k, v)

        if ckpt is not None:
            mlflow.set_tag("resume", "true")
            mlflow.set_tag("resume_from", str(cfg.resume_from))
        if cfg.init_from and ckpt is None:
            mlflow.set_tag("init_from", str(cfg.init_from))

        mlflow_active = True

    writer = None
    if cfg.use_tensorboard and accelerator.is_main_process:
        os.makedirs(cfg.tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=cfg.tb_dir)

    if cfg.epochs is None:
        raise ValueError("cfg.epochs cannot be None for a fresh run (only allowed in resume configs).")
    if start_epoch > cfg.epochs:
        if accelerator.is_main_process:
            print(f"[WARN] resume epoch ({start_epoch}) > cfg.epochs ({cfg.epochs}). Nothing to do.")
        if mlflow_active:
            mlflow.end_run()
        return

    best_path, last_path = checkpoint_paths(cfg.out)
    global_step = (start_epoch - 1) * max(1, len(train_loader))

    for epoch in range(start_epoch, int(cfg.epochs) + 1):
        t0 = time.time()

        avg_loss, global_step = train_one_epoch(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_loader=train_loader,
            accelerator=accelerator,
            device=device,
            epoch=epoch,
            log_steps=int(cfg.log_steps),
            writer=writer,
            global_step_base=global_step,
        )
        dice = validate_dice(
            model=model,
            val_loader=val_loader,
            dice_metric=dice_metric,
            accelerator=accelerator,
            device=device,
            threshold=0.5,
        )

        dt = time.time() - t0
        if writer:
            writer.add_scalar("train/epoch_loss", float(avg_loss), epoch)
            writer.add_scalar("val/dice", float(dice), epoch)
            writer.add_scalar("train/epoch_time_sec", float(dt), epoch)
        if mlflow_active:
            mlflow.log_metric("train/epoch_loss", float(avg_loss), step=epoch)
            mlflow.log_metric("val/dice", float(dice), step=epoch)
            mlflow.log_metric("train/epoch_time_sec", float(dt), step=epoch)

        if accelerator.is_main_process:
            print(f"Epoch {epoch:03d}/{cfg.epochs} | loss={avg_loss:.4f} | val_dice={dice:.4f} | {dt:.1f}s")

        if dice > best_dice and accelerator.is_main_process:
            best_dice = float(dice)
            unwrapped = accelerator.unwrap_model(model)
            torch.save(
                {
                    "model": unwrapped.state_dict(),
                    "optimizer": getattr(optimizer, "optimizer", optimizer).state_dict(),
                    "epoch": epoch,
                    "best_dice": best_dice,
                    "cfg": cfg.__dict__,
                    "mlflow_run_id": mlflow_run_id,
                },
                best_path,
            )

            if mlflow_active:
                mlflow.log_metric("best/val_dice", float(best_dice), step=epoch)
                mlflow.log_artifact(best_path, artifact_path="checkpoints")

        if accelerator.is_main_process and cfg.save_last_every > 0 and (epoch % int(cfg.save_last_every) == 0):
            unwrapped = accelerator.unwrap_model(model)
            torch.save(
                {
                    "model": unwrapped.state_dict(),
                    "optimizer": getattr(optimizer, "optimizer", optimizer).state_dict(),
                    "epoch": epoch,
                    "best_dice": best_dice,
                    "cfg": cfg.__dict__,
                    "mlflow_run_id": mlflow_run_id,
                },
                last_path,
            )

    if writer:
        writer.close()
    if accelerator.is_main_process:
        print(f"Done. Best val dice: {best_dice:.4f}")
    if mlflow_active:
        mlflow.end_run()
