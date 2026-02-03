# UW-Madison GI Tract Segmentation (MONAI 3D)

This repo follows the same structure as `/home/jeremiah/github/Guqin_Image_Generation/`:
- `configs/` for YAML configuration
- `scripts/` for run wrappers
- `src/uwgi/` for training and inference modules
- `outputs/` for checkpoints and inference outputs

## Setup

Recommended: Python 3.10/3.11.

Install dependencies (example):
```bash
pip install torch monai opencv-python pandas numpy pyyaml accelerate tensorboard
```

## Training

Edit `configs/train.yaml` (set `run_dir`), then:
```bash
bash scripts/run_train.sh configs/train.yaml --mlflow_run_name 3dunet-baseline
```

Which config is used?
- `bash scripts/run_train.sh <CONFIG>.yaml` uses the YAML you pass in.
- If you run `bash scripts/run_train.sh` without args, it defaults to `configs/train.yaml`.

Notes:
- Default model is `swin_unetr`. For SwinUNETR, `patch_d/patch_h/patch_w` should be divisible by 32.
- `mixed_precision` can be `no`, `fp16`, or `bf16` (requires `accelerate`).
- `use_tensorboard` enables logging to `tb_dir`.
- You can provide explicit train/val splits via `train_ids` and `val_ids` (csv/txt/json); this overrides `val_ratio`.
- MLflow params/metrics are logged if `use_mlflow: true`. This repo pins MLflow to `mlflow.db` by default; start UI with:
```bash
mlflow ui --backend-store-uri sqlite:///$(pwd)/mlflow.db --host 0.0.0.0 --port 5000
```

## SwinUNETR pretrained init (BTCV)

This repo supports loading a SwinUNETR pretrained checkpoint as an initialization (partial load by shape match).

```bash
bash scripts/download_swin_unetr_btcv.sh
bash scripts/run_train.sh configs/train_swin_unetr_btcv_init.yaml
```

## Preset configs (fp16, 30 epochs)

- UNet: `configs/train_unet_fp16_16gb.yaml`
- SwinUNETR: `configs/train_swin_unetr_fp16_16gb.yaml`

## Resume training (checkpoint + MLflow resume)

This repo writes:
- `best.pt` when validation improves
- `last.pt` every epoch (configurable via `save_last_every`)

To resume in-place (same `run_dir`, same MLflow `run_id`, continuous curves), set `resume_from` to a previous run's `last.pt` (or pass it via CLI) using `configs/train_resume.yaml`, then:
```bash
bash scripts/run_train.sh configs/train_resume.yaml --resume_from /path/to/last.pt
```

## Inference (submission)

Edit `configs/infer.yaml`, then:
```bash
bash scripts/run_infer.sh configs/infer.yaml
```

Notes:
- `uwgi.infer_cli` expects a Kaggle-style `test/` directory under `data_root`.
- If `sample_submission` is provided, outputs are merged to match Kaggle ordering.

## Validation (visualization)

Use `val_pipeline.ipynb` to pick and visualize a small set of validation slices (GT vs Pred) for qualitative checks.

During training, you can also log a fixed set of val slice visualizations to TensorBoard via:
- `val_vis_every` (0 disables)
- `val_vis_n` (default 16)
- `val_vis_samples` (JSON path, default: `/home/jeremiah/github/UW-Madison_GI_Tract_Image_Segmentation/inputs/splits/val_vis_samples.json`)

For quick GT visualization in notebooks, see `uwgi.viz.plot_case_day_slice_gt`.
For GT vs prediction visualization, see `uwgi.viz.plot_case_day_slice_gt_pred`.


mlflow ui --backend-store-uri sqlite:///$(pwd)/mlflow.db --host 0.0.0.0 --port 5000
