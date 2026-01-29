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

Notes:
- Default model is `swin_unetr`. For SwinUNETR, `patch_d/patch_h/patch_w` should be divisible by 32.
- `mixed_precision` can be `no`, `fp16`, or `bf16` (requires `accelerate`).
- `use_tensorboard` enables logging to `tb_dir`.
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


mlflow ui --backend-store-uri sqlite:///$(pwd)/mlflow.db --host 0.0.0.0 --port 5000
