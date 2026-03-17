from types import SimpleNamespace

import numpy as np
import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    OneOf,
    RandAffined,
    RandCoarseDropoutd,
    RandFlipd,
    RandGridDistortiond,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    SpatialPadd,
)

from src.datasets import LoadCaseDayVolumed


def _safe_unit_scale(x):
    max_value = float(np.max(x))
    if max_value <= 0.0:
        return x.astype(np.float32)
    return (x / max_value).astype(np.float32)


cfg = SimpleNamespace(**{})

# data
cfg.data_root = "inputs"
cfg.train_ids = "inputs/splits/train_case_days.csv"
cfg.val_ids = "inputs/splits/val_case_days.csv"
cfg.num_workers = 6
cfg.seed = 42

# output / resume
cfg.output_dir = ""
cfg.resume_from = ""

# tracking (MLflow)
cfg.use_mlflow = True
cfg.allow_dirty_git = True
cfg.mlflow_tracking_uri = "sqlite:///./mlflow.db"
cfg.mlflow_experiment = "uwgi/segmentation"

# model
cfg.model_name = "Unet3D_ref_aug_epoch_150"
cfg.spatial_dims = 3
cfg.in_channels = 1
cfg.out_channels = 3

# patches / transforms
# Current training pipeline uses (D, H, W).
cfg.patch_d = 80
cfg.patch_h = 160
cfg.patch_w = 160

# optimizer
cfg.lr = 1.0e-4
cfg.weight_decay = 1.0e-4

# OneCycleLR
cfg.lr_max = 5.0e-4
cfg.lr_pct_start = 0.2
cfg.lr_div_factor = 25
cfg.lr_final_div_factor = 1000

# training
cfg.epochs = 150
cfg.val_interval = 5
cfg.train_patch_metrics_interval = 5
cfg.val_hd95_interval = 0
cfg.mixed_precision = "fp16"
cfg.train_batch_size = 8
cfg.val_batch_size = 1

# val (sliding window)
cfg.sw_batch_size = 4
cfg.sw_overlap = 0.25

# visualization (TensorBoard images)
cfg.val_vis_every = 10
cfg.val_vis_samples = "inputs/splits/val_vis_samples.json"

# caching
cfg.train_cache_rate = 0.8
cfg.val_cache_rate = 0.0


def build_train_transforms(*, cfg, case_day_slices, rle_index):
    return Compose(
        [
            LoadCaseDayVolumed(
                keys=["case_day"],
                case_day_slices=case_day_slices,
                rle_index=rle_index,
            ),
            EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            Lambdad(keys=["image"], func=_safe_unit_scale),
            SpatialPadd(keys=["image", "label"], spatial_size=cfg.patch_size, mode="constant"),
            RandSpatialCropd(
                keys=["image", "label"],
                roi_size=cfg.patch_size,
                random_size=False,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[1]),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[2]),
            RandAffined(
                keys=["image", "label"],
                prob=0.5,
                rotate_range=(0.0, 0.0, float(np.pi / 12.0)),
                translate_range=(0.0, cfg.patch_h * 0.0625, cfg.patch_w * 0.0625),
                scale_range=(0.0, 0.1, 0.1),
                mode="nearest",
                padding_mode="reflection",
            ),
            OneOf(
                [
                    RandGridDistortiond(
                        keys=["image", "label"],
                        prob=0.5,
                        distort_limit=(-0.05, 0.05),
                        mode="nearest",
                        padding_mode="reflection",
                    ),
                    RandCoarseDropoutd(
                        keys=["image", "label"],
                        holes=5,
                        max_holes=8,
                        spatial_size=(1, 1, 1),
                        max_spatial_size=(12, 12, 12),
                        fill_value=0.0,
                        prob=0.5,
                    ),
                ]
            ),
            RandScaleIntensityd(keys=["image"], factors=(-0.2, 0.2), prob=0.5),
            RandShiftIntensityd(keys=["image"], offsets=(-0.1, 0.1), prob=0.5),
            EnsureTyped(keys=["image", "label"], dtype=torch.float32),
        ]
    )


def build_val_transforms(*, cfg, case_day_slices, rle_index):
    return Compose(
        [
            LoadCaseDayVolumed(
                keys=["case_day"],
                case_day_slices=case_day_slices,
                rle_index=rle_index,
            ),
            EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            Lambdad(keys=["image"], func=_safe_unit_scale),
            EnsureTyped(keys=["image", "label"], dtype=torch.float32),
        ]
    )
