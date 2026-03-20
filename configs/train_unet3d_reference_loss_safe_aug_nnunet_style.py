from types import SimpleNamespace

import numpy as np
import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    OneOf,
    RandAdjustContrastd,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    RandZoomd,
    SpatialPadd,
)

from src.datasets import LoadCaseDayVolumed
from src.losses import DiceBceMultilabelLoss


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
cfg.model_name = "Unet3D_ref_loss_diceBCE_safe_aug_nnunet_style"
cfg.spatial_dims = 3
cfg.in_channels = 1
cfg.out_channels = 3

# patches / transforms
cfg.patch_d = 96
cfg.patch_h = 224
cfg.patch_w = 224

# optimizer
cfg.lr = 1.0e-4
cfg.weight_decay = 1.0e-4

# loss
cfg.loss = DiceBceMultilabelLoss(w_dice=0.5, w_bce=0.5)

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
                prob=0.35,
                rotate_range=(0.0, 0.0, float(np.pi / 12.0)),
                translate_range=(0.0, 12.0, 12.0),
                scale_range=(0.0, 0.08, 0.08),
                mode=("bilinear", "nearest"),
                padding_mode=("border", "border"),
            ),
            # Add nnU-Net-style intensity augmentation on top of the safer spatial recipe.
            RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.05),
            RandGaussianSmoothd(keys=["image"], prob=0.2, sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0)),
            RandScaleIntensityd(keys=["image"], factors=(-0.15, 0.15), prob=0.3),
            RandAdjustContrastd(keys=["image"], prob=0.15, gamma=(0.75, 1.25)),
            OneOf(
                transforms=[
                    RandAdjustContrastd(keys=["image"], prob=1.0, gamma=(0.7, 1.5), invert_image=False),
                    RandAdjustContrastd(keys=["image"], prob=1.0, gamma=(0.7, 1.5), invert_image=True),
                ],
                weights=[0.75, 0.25],
            ),
            RandZoomd(
                keys=["image", "label"],
                prob=0.15,
                min_zoom=(0.7, 0.7, 0.7),
                max_zoom=(1.0, 1.0, 1.0),
                mode=("trilinear", "nearest"),
                padding_mode=("edge", "edge"),
                keep_size=True,
            ),
            RandShiftIntensityd(keys=["image"], offsets=(-0.08, 0.08), prob=0.3),
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
