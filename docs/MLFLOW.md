# MLflow Tracking (Training)

This repo logs training runs to MLflow from `scripts/train_unet3d.py` when `use_mlflow: true` in your config.

## Quick start

1) Install MLflow:
```bash
pip install mlflow
```

2) Enable tracking in `configs/train_unet3d.yaml`:
```yaml
use_mlflow: true
mlflow_tracking_uri: "sqlite:///./mlflow.db"   # optional
mlflow_experiment: "uwgi/segmentation"         # optional
```

3) Run training:
```bash
python scripts/train_unet3d.py --config configs/train_unet3d.yaml
```

4) Open the UI (SQLite backend in this repo):
```bash
mlflow ui --backend-store-uri sqlite:///$(pwd)/mlflow.db --host 0.0.0.0 --port 5000
```

## Where MLflow data lives (default)

- Tracking backend:
  - Uses `mlflow_tracking_uri` from the YAML when set.
  - Otherwise uses `$MLFLOW_TRACKING_URI` when set.
  - Otherwise uses `sqlite:///<repo>/mlflow.db` when `mlflow.db` exists.
- Artifacts:
  - By default MLflow writes artifacts to `mlruns/` under the current working directory.
  - `scripts/train_unet3d.py` changes the working directory to the repo root, so you should see `mlruns/` in the repo root unless you override MLflow defaults.

## What gets logged

### Params

- All keys from the resolved training config (`TrainCfg` in `scripts/train_unet3d.py`) are logged as MLflow params (flat key/value).

### Tags

- Git (when available):
  - `git.commit`, `git.branch`, `git.describe`, `git.dirty`
  - If `allow_dirty_git: true` and the repo is dirty: `git.dirty_allowed=true`
- Environment:
  - `python`, `platform`
  - Best-effort: `torch`, `monai`, `accelerate`, plus CUDA/GPU tags (`cuda.available`, `cuda.version`, `cudnn.version`, `gpu.name`)
- Reproducibility / linkage:
  - `run_dir`: the local output directory for this run (under `outputs/` unless you override `output_dir`)
  - `config.path`, `config.sha256` (the YAML used)
  - `split.train.path`, `split.train.sha256`, `split.val.path`, `split.val.sha256` (when split files exist)
  - `summary/best_metric_epoch` (written at the end of training)

### Metrics (logged per epoch)

- Training:
  - `train/loss`, `train/lr`, `train/grad_norm`
  - Patch Dice: `train/dice_patch_mean` and `train/dice_patch_<class>` for each class in `src/constants.py`
- Validation (on `val_interval` epochs):
  - `val/dice_sw_mean`, `val/dice_sw_<class>` for each class
  - `val/loss_sw`
- Timing / system:
  - `system/train_time_sec`, `system/val_time_sec`, `system/total_training_time_hr`
  - CUDA memory (when CUDA is available): `system/gpu_mem_allocated_gb`, `system/gpu_mem_reserved_gb`, `system/gpu_mem_max_allocated_gb`
  - NVML stats (when `pynvml` is installed and available): `system/gpu_utilization_pct`, `system/gpu_mem_utilization_pct`, `system/gpu_mem_used_gb`
- Summary (end of training):
  - `summary/best_metric` (best `val/dice_sw_mean`)
  - If NVML stats were collected: `summary/avg_gpu_utilization_pct`, `summary/avg_gpu_mem_utilization_pct`

### Artifacts

Logged at the end of training:
- `run/`:
  - `config.yaml`
  - `config.resolved.json`
  - `train.log`
- `checkpoints/`:
  - `best.pt` (best val)
  - `last.pt` (last epoch)
- `tb/`:
  - TensorBoard event files (the entire `outputs/<run>/tb/` directory)

Logged early (so the run is self-contained even if training crashes later):
- `splits/`: the configured split files, when they exist (`train_ids`, `val_ids`)

If the repo is dirty and `allow_dirty_git: true`:
- `git/`: `git_status_porcelain.txt`, `git_diff.patch`, and (when non-empty) `git_diff_cached.patch`

## Clean git requirement (optional)

When `use_mlflow: true`:
- If `allow_dirty_git: false` and `git status --porcelain` is non-empty, training exits before starting.
- If `allow_dirty_git: true`, training proceeds and logs a snapshot of local changes as MLflow artifacts (best-effort).

## Resuming a run (checkpoint + MLflow)

Each training output directory writes `mlflow_run_id.txt` containing the MLflow `run_id` used for that run.

To resume and keep logging into the same MLflow run:
1) Set `output_dir` to the original run directory.
2) Set `resume_from` to that directory’s `last.pt`.
3) Keep `use_mlflow: true`, then run training again.

Example:
```yaml
output_dir: "outputs/20260305-120000_Unet3D"
resume_from: "outputs/20260305-120000_Unet3D/last.pt"
use_mlflow: true
```

If `resume_from` is empty, or `mlflow_run_id.txt` is missing next to the checkpoint, a new MLflow run is created.

