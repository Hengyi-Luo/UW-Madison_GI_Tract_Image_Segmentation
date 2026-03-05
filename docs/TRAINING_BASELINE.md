# Training Baseline (MONAI 3D U-Net)

This document describes the **current training baseline** implemented in `scripts/train_unet3d.py`. It focuses on what the code does today (data loading, preprocessing, model, loss, training/validation, outputs, and how to run/resume).

## Entry point

- Script: `scripts/train_unet3d.py`
- Config: YAML passed via `--config` (default: `configs/train_unet3d.yaml`)

Run:
```bash
python scripts/train_unet3d.py --config configs/train_unet3d.yaml
```

Notes:
- The script sets the working directory to the repo root at startup, so relative paths in configs are interpreted from the repo root.
- The script currently supports **only one CLI flag**: `--config`. To change things like `resume_from`, edit the YAML.

## Data layout and expectations

The baseline expects a Kaggle-style UW-Madison GI dataset under `data_root` (default: `inputs/`):

```
inputs/
  train.csv
  train/
    caseXXX/
      caseXXX_dayY/
        scans/
          slice_0001_<H>_<W>_<psx>_<psy>.png
          slice_0002_<H>_<W>_<psx>_<psy>.png
          ...
  splits/
    train_case_days.csv
    val_case_days.csv
    val_vis_samples.json   (optional, for visualization)
```

### Split files (`train_ids`, `val_ids`)

- `train_ids` and `val_ids` are CSV files with a header row and a single column `case_day`.
- Example (`inputs/splits/train_case_days.csv`):
  ```
  case_day
  case101_day20
  case101_day22
  ...
  ```
- The script checks that train/val case_day sets do **not** overlap and raises an error if they do.

### Labels (`train.csv`)

- `train.csv` is the Kaggle training CSV with columns:
  - `id` like `case123_day20_slice_0001`
  - `class` in `{large_bowel, small_bowel, stomach}`
  - `segmentation` as RLE (may be empty / NaN for no mask)
- The baseline parses `id` into:
  - `case_day` = `case123_day20`
  - `slice_idx` = `1` (from the last token)

Class/channel order:
- The three foreground classes are defined in `src/constants.py` as:
  `["large_bowel", "small_bowel", "stomach"]`.
- The label tensor uses this exact channel order, and the model’s `out_channels` should match it (typically `3`).

## How a training sample is built

Each dataset item corresponds to one `case_day` (a 3D volume).

### Volume construction

`src/data_utils.build_case_day_slices()` indexes all PNG slice paths per `case_day` by scanning:
`<data_root>/train/<case>/<case_day>/scans/*.png` and sorting by the slice index embedded in the filename.

`src/data_utils.build_rle_index()` builds a dictionary mapping:
`(case_day, slice_idx) -> {class_name: rle_string}` from `train.csv`.

`src/datasets.LoadCaseDayVolumed` then:
- Reads all slices for that `case_day` with OpenCV (`cv2.IMREAD_UNCHANGED`)
- Builds:
  - `image`: `(D, H, W)` (dtype from PNG, often `uint16`)
  - `label`: `(C, D, H, W)` where `C=3` for the three classes (dtype `uint8` in `{0,1}`)
- Decodes RLE masks via `src/rle.rle_decode()` for each `(case_day, slice_idx, class)`
- Also stores:
  - `orig_shape`: `(D, H, W)`
  - `slice_idxs`: list of the Kaggle slice indices corresponding to each depth position `z`

## Preprocessing and augmentation (MONAI transforms)

Transforms are defined in `_build_transforms()` in `scripts/train_unet3d.py`.

Common (train + val):
- `LoadCaseDayVolumed(...)` loads full 3D volume and 3D masks from disk + `train.csv`.
- `EnsureChannelFirstd(keys=["image"], channel_dim="no_channel")` converts image `(D,H,W)` to `(1,D,H,W)`.
- `ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0)` scales image intensities to `[0, 1]`.
- `SpatialPadd(keys=["image","label"], spatial_size=(patch_d,patch_h,patch_w))` pads smaller volumes to at least the patch size.

Training-only:
- `RandSpatialCropd(keys=["image","label"], roi_size=patch_size, random_size=False)` samples a fixed-size 3D patch.

Validation:
- No cropping; validation runs on the full (padded) volume using sliding-window inference.

## Model

The baseline uses MONAI’s `Unet` (3D) from `monai.networks.nets`:

- `spatial_dims=3`
- `in_channels=1`
- `out_channels=3` (one sigmoid output per class)
- Encoder channels: `(32, 64, 128, 256, 512)`
- Strides: `(2, 2, 2, 2)`
- `num_res_units=2`
- `dropout=0.2`
- `norm=BatchNorm`

Important implication:
- Outputs are **multi-label** logits with shape `(B, 3, D, H, W)`.
- Post-processing uses `sigmoid` + threshold `0.5` per class (not softmax).

## Loss

Loss is `DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")`.

- `sigmoid=True` matches the multi-label setup.
- The Dice + CE combination helps class imbalance and stabilizes training.

## Optimization and LR schedule

- Optimizer: `AdamW(lr=cfg.lr, weight_decay=cfg.weight_decay)`
- LR schedule: `OneCycleLR`
  - `max_lr=cfg.lr_max`
  - `pct_start=cfg.lr_pct_start`
  - `div_factor=cfg.lr_div_factor`
  - `final_div_factor=cfg.lr_final_div_factor`
  - `steps_per_epoch=len(train_loader)`
  - Stepped **every iteration** (each batch) via `scheduler.step()`

## Mixed precision and device handling (Accelerate)

The script uses `accelerate.Accelerator(mixed_precision=cfg.mixed_precision)` and wraps forward passes in `accelerator.autocast()`.

Supported values:
- `mixed_precision: "no" | "fp16" | "bf16"`

The model/optimizer/dataloaders are passed through `accelerator.prepare(...)`.

## Training loop

For each epoch:
1) `train_one_epoch()`:
   - Forward + loss under autocast
   - `accelerator.backward(loss)`
   - Gradient norm is computed via `clip_grad_norm_(..., max_norm=inf)` (used only for logging; no clipping)
   - `optimizer.step()` then `scheduler.step()`
2) Logs training scalars (TensorBoard + optional MLflow):
   - `train/loss`, `train/lr`, `train/grad_norm`, `system/train_time_sec`
   - CUDA memory stats when CUDA is available
   - Optional NVML GPU utilization stats when `pynvml` is installed
3) Every `val_interval` epochs:
   - Computes train “patch dice” over the training loader (on random patches)
   - Computes validation metrics over full volumes using sliding-window inference
   - Saves `best.pt` if validation Dice improves
4) Always saves `last.pt` at the end of the epoch

## Validation and metrics

### Patch Dice (train)

`compute_patch_dice()` runs the model on the **training loader** and computes Dice on the randomly cropped patches:
- Prediction: `sigmoid(logits) > 0.5`
- Metric: `monai.metrics.DiceMetric(reduction="mean_batch")`
- Logged as:
  - `train/dice_patch_mean`
  - `train/dice_patch_<class>` per class

This metric is useful as a quick sanity check but is **not** directly comparable to full-volume validation Dice.

### Sliding-window Dice (val)

`compute_sw_dice()` runs on the validation loader with `monai.inferers.sliding_window_inference(...)`:
- `roi_size = (patch_d, patch_h, patch_w)`
- `sw_batch_size = cfg.sw_batch_size`
- `overlap = cfg.sw_overlap`
- Prediction: `sigmoid(logits) > 0.5`
- Logs:
  - `val/dice_sw_mean`
  - `val/dice_sw_<class>` per class
  - `val/loss_sw`
  - `system/val_time_sec`

The best model selection uses `val/dice_sw_mean`.

## Validation visualizations (TensorBoard images)

If `val_vis_every > 0` and `val_vis_samples` can be loaded, the script logs a small fixed set of 2D slice visualizations to TensorBoard at epochs divisible by `val_vis_every`.

- File: `val_vis_samples` is expected to be JSON with a `samples` list. Each sample should contain:
  - `case_day` (e.g. `case101_day20`)
  - `slice_pos` (0-indexed depth position in the volume; **not** Kaggle slice index)

Generation helper:
- `scripts/make_vis_samples.py` can generate such JSON/CSV sample lists from a split file.

TensorBoard image tags:
- `val_vis/<case_day>/slice_<NNN>/raw`
- `val_vis/<case_day>/slice_<NNN>/gt`
- `val_vis/<case_day>/slice_<NNN>/pred`

Implementation detail:
- `raw` and `gt` are logged once per tag (static); `pred` can update over time.

## Outputs and run directory structure

Each run writes to `output_dir`:
- If `output_dir` is empty, it defaults to `outputs/<timestamp>_<model_name>/`.

Typical contents:
```
outputs/<run>/
  config.yaml
  config.resolved.json
  train.log
  tb/                       # TensorBoard event files
  best.pt                   # best by val/dice_sw_mean
  last.pt                   # last epoch
  mlflow_run_id.txt          # only when use_mlflow=true
  git_*.patch / git_status*  # only when use_mlflow=true and allow_dirty_git=true (best-effort)
```

## Checkpoint format

`best.pt` contains:
- `epoch`
- `model_state_dict`
- `optimizer_state_dict`
- `scheduler_state_dict`
- `best_metric`, `best_metric_epoch`
- `dice_per_class` (validation per-class Dice at the best epoch)

`last.pt` contains:
- `epoch` (next epoch to run, i.e. `epoch+1`)
- `model_state_dict`
- `optimizer_state_dict`
- `scheduler_state_dict`
- `best_metric`, `best_metric_epoch`

## Resuming training

To resume weights + optimizer + scheduler:
- Set `resume_from` in the YAML to a checkpoint path (usually `.../last.pt`).

To resume **in place** (keep writing into the same output directory), also set:
- `output_dir` to the original run directory.

Example:
```yaml
output_dir: "outputs/20260305-120000_Unet3D"
resume_from: "outputs/20260305-120000_Unet3D/last.pt"
```

## Experiment tracking (TensorBoard + MLflow)

### TensorBoard

- TensorBoard logs are always written under `outputs/<run>/tb/`.
- Start TensorBoard (example):
  ```bash
  tensorboard --logdir outputs --port 6006 --host 0.0.0.0
  ```

### MLflow (optional)

- Controlled by `use_mlflow` in the YAML.
- When enabled, params/metrics/artifacts are logged (including TensorBoard event files).
- Details are documented in `docs/MLFLOW.md`.

## Configuration reference (current keys)

All YAML keys must match fields in `TrainCfg` (unknown keys raise an error). Current groups:

- Data:
  - `data_root`, `train_ids`, `val_ids`, `num_workers`, `seed`
- Output / resume:
  - `output_dir`, `resume_from`
- Tracking (MLflow):
  - `use_mlflow`, `allow_dirty_git`, `mlflow_experiment`, `mlflow_tracking_uri`
- Model:
  - `model_name`, `spatial_dims`, `in_channels`, `out_channels`
- Patch / cropping:
  - `patch_d`, `patch_h`, `patch_w`
- Optimizer:
  - `lr`, `weight_decay`
- OneCycleLR:
  - `lr_max`, `lr_pct_start`, `lr_div_factor`, `lr_final_div_factor`
- Training:
  - `epochs`, `val_interval`, `mixed_precision`, `train_batch_size`, `val_batch_size`
- Validation (sliding window):
  - `sw_batch_size`, `sw_overlap`
- Visualization:
  - `val_vis_every`, `val_vis_samples`
- Caching:
  - `train_cache_rate`, `val_cache_rate`

See example configs:
- `configs/train_unet3d.yaml`

## Common pitfalls / troubleshooting

- **KeyError for a case_day**: the split CSV may contain `case_day` values not present under `<data_root>/train/`. Ensure the directory structure matches Kaggle’s layout.
- **Train/val overlap error**: the script refuses to run if the same `case_day` appears in both split files.
- **Out-of-memory during caching**: `CacheDataset` can be RAM-heavy because each item is a full 3D volume + masks. Reduce `train_cache_rate` / `val_cache_rate` or `num_workers`.
- **Patch size constraints**: `patch_d/patch_h/patch_w` must be compatible with your GPU memory. Larger patches increase both training memory and validation SW inference cost.

