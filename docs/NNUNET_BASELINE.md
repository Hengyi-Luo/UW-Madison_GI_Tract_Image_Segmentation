# nnUNet Baseline

This repo now includes an `nnUNet v2` baseline that follows the official `nnUNet` dataset and training flow while keeping tracking and auxiliary evaluation inside this project.

Default baseline choices:

- planner: `nnUNetPlannerResEncL`
- plans: `nnUNetResEncUNetPlans`
- trainer: `nnUNetTrainer`
- configuration: `3d_fullres`
- fold: `0`

## Installation

Official `nnUNet` installation guidance requires installing PyTorch first, then `nnunetv2`.

If you use this repo environment file, `torch` and `nnunetv2` are both included:

```bash
conda env create -f environment.yml
conda activate uwgi
```

If you are installing manually, follow this order:

```bash
pip install torch
pip install nnunetv2
```

`nnUNet` commands must be on your `PATH`:

- `nnUNetv2_plan_and_preprocess`
- `nnUNetv2_train`

## Data Preparation

Prepare the raw `nnUNet` dataset:

```bash
python scripts/prepare_nnunet_dataset.py --config configs/train_nnunet_baseline.py
```

This generates:

- `nnUNet_raw/Dataset501_UWGI/imagesTr/<case_day>_0000.nii.gz`
- `nnUNet_raw/Dataset501_UWGI/labelsTr/<case_day>.nii.gz`
- `nnUNet_raw/Dataset501_UWGI/dataset.json`
- `nnUNet_raw/Dataset501_UWGI/label_collapse_summary.json`

## Training

Run the full baseline:

```bash
python scripts/train_nnunet_baseline.py --config configs/train_nnunet_baseline.py
```

The wrapper will:

1. prepare the `nnUNet_raw` dataset
2. run `nnUNetv2_plan_and_preprocess`
3. write `nnUNet_preprocessed/Dataset501_UWGI/splits_final.json`
4. run `nnUNetv2_train 501 3d_fullres 0 -p nnUNetResEncUNetPlans`
5. collect the official validation summary
6. run repo-side auxiliary evaluation and export `submit.csv`
7. log metrics and artifacts to MLflow

## MLflow

The wrapper follows the same tracking backend pattern as the MONAI training script.

Open the UI with:

```bash
mlflow ui --backend-store-uri sqlite:///$(pwd)/mlflow.db --host 0.0.0.0 --port 5000
```

The run logs:

- config and split hashes
- `dataset.json`, `splits_final.json`, and `label_collapse_summary.json`
- official `nnUNet` validation summary
- repo-side `eval_summary.json`, `eval_per_case.csv`, and `submit.csv`
- timing metrics for prepare, preprocess, train, and eval

## Label Caveat

This task is originally represented as 3-channel multilabel segmentation.

`nnUNet` standard training requires integer label maps, so this baseline uses a lossy collapse into:

- `0`: background
- `1`: `large_bowel`
- `2`: `small_bowel`
- `3`: `stomach`

When overlaps exist, labels are written in this fixed order:

- `large_bowel`
- `small_bowel`
- `stomach`

Later labels overwrite earlier labels, so the effective priority is:

- `stomach > small_bowel > large_bowel`

This is documented in `label_collapse_summary.json` and logged to MLflow.

## Metrics

Two validation views are kept:

- official `nnUNet` validation from `validation/summary.json`
- repo-side auxiliary evaluation under `outputs/<run>/eval/`

These numbers are useful together, but they are not identical metrics and should not be interpreted as interchangeable.
