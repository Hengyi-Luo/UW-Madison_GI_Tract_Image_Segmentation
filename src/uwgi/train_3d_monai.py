from __future__ import annotations

import json
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
from .data_utils import build_case_day_slices, build_rle_index, load_case_days
from .datasets import UWGI3DPatchDataset
from .datasets import UWGI3DFullVolumeDataset
from .engine import train_one_epoch, validate_dice, validate_sliding_window_dice
from .experiment import git_info, resolve_run_naming
from .models import build_model
from .splits import split_case_days


def _load_val_vis_samples(path: str) -> list[dict]:
    """
    Load a fixed set of val visualization samples from JSON.

    Supported formats:
      - list[dict] (legacy)
      - {"samples": list[dict], ...} (recommended)

    Each sample must contain:
      - case_day: str
      - slice_pos: int  (index into case_day_slices[case_day])
    Optional:
      - slice_idx: int  (for nicer TB tags)
    """
    if not path:
        return []
    if not os.path.exists(path):
        print(f"[WARN] val_vis_samples not found: {path}")
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read val_vis_samples: {path} ({e})")
        return []

    if isinstance(obj, dict) and isinstance(obj.get("samples"), list):
        samples = obj["samples"]
    elif isinstance(obj, list):
        samples = obj
    else:
        print(f"[WARN] Unrecognized val_vis_samples JSON shape: {type(obj)}")
        return []

    out: list[dict] = []
    for s in samples:
        if not isinstance(s, dict):
            continue
        if "case_day" not in s or "slice_pos" not in s:
            continue
        out.append(s)
    return out


def _log_val_samples_to_tb(
    *,
    writer: SummaryWriter,
    epoch: int,
    samples: list[dict],
    case_day_slices: dict[str, list[str]],
    rle_index,
    model,
    roi_size: tuple[int, int, int],
    sw_batch_size: int,
    overlap: float,
    threshold: float,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] Skipping TB val sample plots (matplotlib not available): {e}")
        return

    from .viz import plot_case_day_slice_gt_pred

    model_for_viz = model

    for i, s in enumerate(samples):
        case_day = str(s["case_day"])
        slice_pos = int(s["slice_pos"])
        slice_idx = int(s.get("slice_idx", -1))
        tag_base = f"val_samples/{i:02d}_{case_day}_slice_{slice_idx:04d}"

        fig_grid, fig_overlay, _, _, _ = plot_case_day_slice_gt_pred(
            case_day,
            slice_pos=slice_pos,
            case_day_slices=case_day_slices,
            rle_index=rle_index,
            model=model_for_viz,
            mode="sw",
            roi_size=roi_size,
            sw_batch_size=int(sw_batch_size),
            overlap=float(overlap),
            threshold=float(threshold),
        )
        writer.add_figure(f"{tag_base}/grid", fig_grid, global_step=epoch, close=True)
        writer.add_figure(f"{tag_base}/overlay", fig_overlay, global_step=epoch, close=True)
        plt.close("all")


@dataclass
class TrainCfg:
    # Data
    data_root: str = "./input/uw-madison-gi-tract-image-segmentation"
    seed: int = 42
    val_ratio: float = 0.2
    train_ids: str = ""  # optional explicit train case_day list file (csv/txt/json)
    val_ids: str = ""  # optional explicit val case_day list file (csv/txt/json)
    ids_column: str = ""  # optional csv column name for train_ids/val_ids
    num_workers: int = 4
    cache_in_ram: bool = False

    # Patches
    patch_d: int = 96
    patch_h: int = 224
    patch_w: int = 224
    samples_per_volume: int = 12
    train_fg_ratio: float = 0.0
    train_fg_min_voxels: int = 1
    train_fg_jitter: int = 0

    # Validation
    val_sw_every: int = 0  # 0 = disabled (patch-only); otherwise run full-volume SW val every N epochs
    val_sw_patch_d: int | None = None
    val_sw_patch_h: int | None = None
    val_sw_patch_w: int | None = None
    val_sw_batch_size: int = 1
    val_sw_overlap: float = 0.25
    best_metric: str = "patch"  # "patch" | "sw"

    # TensorBoard val sample visualization (full-slice GT vs Pred)
    val_vis_every: int = 0  # 0 = disabled; otherwise log every N epochs
    val_vis_n: int = 16
    val_vis_threshold: float = 0.5
    val_vis_samples: str = "./input/uw-madison-gi-tract-image-segmentation/splits/val_vis_samples.json"

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

    if cfg.train_ids or cfg.val_ids:
        col = str(cfg.ids_column).strip() or None
        train_set = set(load_case_days(cfg.train_ids, column=col)) if cfg.train_ids else set()
        val_set = set(load_case_days(cfg.val_ids, column=col)) if cfg.val_ids else set()

        missing_train = sorted([cd for cd in train_set if cd not in case_day_slices])
        missing_val = sorted([cd for cd in val_set if cd not in case_day_slices])
        if missing_train:
            print(f"[WARN] train_ids missing from train/ directory: {len(missing_train)} (e.g. {missing_train[:5]})")
        if missing_val:
            print(f"[WARN] val_ids missing from train/ directory: {len(missing_val)} (e.g. {missing_val[:5]})")
        train_set = {cd for cd in train_set if cd in case_day_slices}
        val_set = {cd for cd in val_set if cd in case_day_slices}

        if cfg.train_ids and cfg.val_ids:
            overlap = sorted(train_set & val_set)
            if overlap:
                raise ValueError(f"train_ids and val_ids overlap (n={len(overlap)}), e.g. {overlap[:5]}")
            train_days = sorted(train_set)
            val_days = sorted(val_set)
        elif cfg.val_ids:
            val_days = [cd for cd in all_case_days if cd in val_set]
            train_days = [cd for cd in all_case_days if cd not in val_set]
        else:
            train_days = [cd for cd in all_case_days if cd in train_set]
            val_days = [cd for cd in all_case_days if cd not in train_set]

        if not train_days:
            raise ValueError("No train case_days after applying train_ids/val_ids filtering.")
        if not val_days:
            raise ValueError("No val case_days after applying train_ids/val_ids filtering.")
    else:
        train_days, val_days = split_case_days(all_case_days, val_ratio=cfg.val_ratio, seed=cfg.seed)
    print(f"Train volumes: {len(train_days)} | Val volumes: {len(val_days)}")

    split_dir = cfg.run_dir or os.path.dirname(cfg.out) or "."
    os.makedirs(split_dir, exist_ok=True)
    split_path = os.path.join(split_dir, "split_case_days.json")
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(
            {"seed": int(cfg.seed), "val_ratio": float(cfg.val_ratio), "train": train_days, "val": val_days},
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Saved train/val split: {split_path}")

    # Load a fixed set of val slices for TensorBoard visualization (no in-training sampling).
    val_vis_samples: list[dict] = []
    if int(cfg.val_vis_every) > 0 and cfg.use_tensorboard:
        val_vis_samples = _load_val_vis_samples(str(cfg.val_vis_samples))
        if val_vis_samples:
            val_vis_samples = val_vis_samples[: int(cfg.val_vis_n)]
            print(f"Loaded TB val samples: {len(val_vis_samples)} ({cfg.val_vis_samples})")
        else:
            print(f"[WARN] No TB val samples loaded. Set `val_vis_samples` to a valid JSON file.")

    patch_size = (int(cfg.patch_d), int(cfg.patch_h), int(cfg.patch_w))
    train_ds = UWGI3DPatchDataset(
        case_days=train_days,
        case_day_slices=case_day_slices,
        rle_index=rle_index,
        patch_size=patch_size,
        samples_per_volume=int(cfg.samples_per_volume),
        fg_ratio=float(cfg.train_fg_ratio),
        fg_min_voxels=int(cfg.train_fg_min_voxels),
        fg_jitter=int(cfg.train_fg_jitter),
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
        fg_ratio=0.0,
        cache_in_ram=bool(cfg.cache_in_ram),
        seed=int(cfg.seed) + 999,
        is_train=False,
        verbose_shape_fix=False,
    )
    val_full_loader = None
    if int(cfg.val_sw_every) > 0:
        val_full_ds = UWGI3DFullVolumeDataset(
            case_days=val_days,
            case_day_slices=case_day_slices,
            rle_index=rle_index,
            cache_in_ram=False,
            verbose_shape_fix=False,
        )
        val_full_loader = DataLoader(
            val_full_ds,
            batch_size=1,
            shuffle=False,
            num_workers=int(cfg.num_workers),
            pin_memory=True,
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

    if val_full_loader is None:
        model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)
    else:
        model, optimizer, train_loader, val_loader, val_full_loader = accelerator.prepare(
            model, optimizer, train_loader, val_loader, val_full_loader
        )

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
        dice_patch = validate_dice(
            model=model,
            val_loader=val_loader,
            dice_metric=dice_metric,
            accelerator=accelerator,
            device=device,
            threshold=0.5,
        )
        dice_sw = None
        if val_full_loader is not None and int(cfg.val_sw_every) > 0 and (epoch % int(cfg.val_sw_every) == 0):
            roi_size = (
                int(cfg.val_sw_patch_d or cfg.patch_d),
                int(cfg.val_sw_patch_h or cfg.patch_h),
                int(cfg.val_sw_patch_w or cfg.patch_w),
            )
            dice_sw = validate_sliding_window_dice(
                model=model,
                val_loader=val_full_loader,
                dice_metric=dice_metric,
                accelerator=accelerator,
                device=device,
                roi_size=roi_size,
                sw_batch_size=int(cfg.val_sw_batch_size),
                overlap=float(cfg.val_sw_overlap),
                threshold=0.5,
            )
        else:
            roi_size = (int(cfg.patch_d), int(cfg.patch_h), int(cfg.patch_w))

        dt = time.time() - t0
        if writer:
            writer.add_scalar("train/epoch_loss", float(avg_loss), epoch)
            writer.add_scalar("val/dice_patch", float(dice_patch), epoch)
            if dice_sw is not None:
                writer.add_scalar("val/dice_sw", float(dice_sw), epoch)
            writer.add_scalar("train/epoch_time_sec", float(dt), epoch)

            if (
                val_vis_samples
                and int(cfg.val_vis_every) > 0
                and (epoch % int(cfg.val_vis_every) == 0)
            ):
                unwrapped = accelerator.unwrap_model(model)
                _log_val_samples_to_tb(
                    writer=writer,
                    epoch=epoch,
                    samples=val_vis_samples,
                    case_day_slices=case_day_slices,
                    rle_index=rle_index,
                    model=unwrapped,
                    roi_size=roi_size,
                    sw_batch_size=int(cfg.val_sw_batch_size),
                    overlap=float(cfg.val_sw_overlap),
                    threshold=float(cfg.val_vis_threshold),
                )
        if mlflow_active:
            mlflow.log_metric("train/epoch_loss", float(avg_loss), step=epoch)
            mlflow.log_metric("val/dice_patch", float(dice_patch), step=epoch)
            if dice_sw is not None:
                mlflow.log_metric("val/dice_sw", float(dice_sw), step=epoch)
            mlflow.log_metric("train/epoch_time_sec", float(dt), step=epoch)

        if accelerator.is_main_process:
            msg = f"Epoch {epoch:03d}/{cfg.epochs} | loss={avg_loss:.4f} | val_patch={dice_patch:.4f}"
            if dice_sw is not None:
                msg += f" | val_sw={dice_sw:.4f}"
            msg += f" | {dt:.1f}s"
            print(msg)

        primary = float(dice_patch)
        if str(cfg.best_metric).lower() == "sw" and dice_sw is not None:
            primary = float(dice_sw)

        if primary > best_dice and accelerator.is_main_process:
            best_dice = float(primary)
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
