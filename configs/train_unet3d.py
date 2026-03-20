from types import SimpleNamespace

from monai.transforms import (
    Compose, 
    EnsureChannelFirstd, 
    EnsureTyped, 
    RandSpatialCropd, 
    ScaleIntensityd, 
    SpatialPadd
)

from src.datasets import LoadCaseDayVolumed


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
cfg.model_name = "Unet3D"
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

# OneCycleLR
cfg.lr_max = 5.0e-4
cfg.lr_pct_start = 0.2
cfg.lr_div_factor = 25
cfg.lr_final_div_factor = 1000

# training
cfg.epochs = 150
cfg.val_interval = 5
cfg.val_hd95_interval = 0
cfg.mixed_precision = "fp16"
cfg.train_batch_size = 8
cfg.val_batch_size = 1

# val (sliding window)
cfg.sw_batch_size = 2
cfg.sw_overlap = 0.25

# visualization (TensorBoard images)
cfg.val_vis_every = 10
cfg.val_vis_samples = "inputs/splits/val_vis_samples.json"

# caching
cfg.train_cache_rate = 1.0
cfg.train_eval_cache_rate = 0.0
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
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            SpatialPadd(keys=["image", "label"], spatial_size=cfg.patch_size, mode="constant"),
            RandSpatialCropd(keys=["image", "label"], roi_size=cfg.patch_size, random_size=False),
            EnsureTyped(keys=["image", "label"]),
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
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            SpatialPadd(keys=["image", "label"], spatial_size=cfg.patch_size, mode="constant"),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
