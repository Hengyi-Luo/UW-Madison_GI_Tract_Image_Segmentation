# UW-Madison GI Tract Segmentation (MONAI 3D)

This repo follows the same structure as `/home/jeremiah/github/Guqin_Image_Generation/`:
- `configs/` for YAML configuration
- `scripts/` for run wrappers
- `train_cli.py` and `infer_cli.py` as CLI entry points
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
bash scripts/run_train.sh configs/train.yaml
```

Notes:
- `mixed_precision` can be `no`, `fp16`, or `bf16` (requires `accelerate`).
- `use_tensorboard` enables logging to `tb_dir`.

## Inference (submission)

Edit `configs/infer.yaml`, then:
```bash
bash scripts/run_infer.sh configs/infer.yaml
```

Notes:
- `infer_cli.py` expects a Kaggle-style `test/` directory under `data_root`.
- If `sample_submission` is provided, outputs are merged to match Kaggle ordering.
