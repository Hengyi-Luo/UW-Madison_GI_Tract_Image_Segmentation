# UW-Madison GI Tract Segmentation (MONAI 3D)

Baseline code for the **UW-Madison GI Tract Image Segmentation** task (Kaggle-style dataset), using a MONAI **3D U-Net** with:
- full-volume validation via sliding-window inference
- Kaggle-style RLE CSV export for inference
- TensorBoard logging (always) + optional MLflow tracking

**Task**: given a 3D volume assembled from 2D grayscale PNG slices (`case_day`), predict **multi-label** binary masks for:
`large_bowel`, `small_bowel`, and `stomach`. Inference exports a Kaggle-style RLE CSV with one row per `(case_day, slice, class)`.

## Example (raw / GT / prediction)

Qualitative validation example for `case122_day27`, `slice_069` at training step `1000`:

<table>
  <tr>
    <th>Raw</th>
    <th>Ground truth (GT)</th>
    <th>Prediction</th>
  </tr>
  <tr>
    <td><img src="docs/images/val_vis_case122_day27_slice069_raw.png" width="256"></td>
    <td><img src="docs/images/val_vis_case122_day27_slice069_gt.png" width="256"></td>
    <td><img src="docs/images/val_vis_case122_day27_slice069_pred.png" width="256"></td>
  </tr>
</table>

Overlay colors (channels in `src/constants.py` order):
- Red: `large_bowel`
- Green: `small_bowel`
- Blue: `stomach`

All three PNGs are the same pixel size; if they *look* different in your markdown preview, it’s typically due to how the renderer scales images inside tables. The fixed `width` above forces consistent display size.

Repo layout:
- `configs/`: tracked training config modules and inference config files
- `scripts/`: training and inference entrypoints
- `src/`: dataset utilities, RLE encode/decode, and model constants
- `docs/`: detailed documentation (baseline + MLflow)

## Setup

Recommended: create the environment from `environment.yml`.

```bash
conda env create -f environment.yml
conda activate uwgi
```

If you prefer `mamba`:
```bash
mamba env create -f environment.yml
mamba activate uwgi
```

`environment.yml` now includes both `torch` and `nnunetv2`. If you install manually instead of using the environment file, follow the official order and install PyTorch first, then `nnunetv2`.

## Baseline validation (reference)

Baseline Dice numbers below are from `outputs/20260302-212100_Unet3D/best.pt`, evaluated via `scripts/infer_unet3d.py`
with `threshold=0.5` and sliding-window `roi_size=[96,224,224]`, `sw_batch_size=2`, `overlap=0.25`.

Reference summaries (these are the only files you need to read):
- Val split: `docs/baseline_infer_val_best/eval_summary.json`
- Train split: `docs/baseline_infer_train_best/eval_train_summary.json`

Val split (`inputs/splits/val_case_days.csv`):
- Mean Dice (58 case_days): **0.7745**
- Mean Dice per class (in `src/constants.py` order):
  - `large_bowel`: **0.7739**
  - `small_bowel`: **0.6964**
  - `stomach`: **0.8531**

Train split (`inputs/splits/train_case_days.csv`):
- Mean Dice (216 case_days): **0.9293**
- Mean Dice per class:
  - `large_bowel`: **0.9317**
  - `small_bowel`: **0.9046**
  - `stomach`: **0.9516**

## Inference benchmark record

Current default inference config:

```bash
python scripts/infer_unet3d.py
```

This now defaults to:

```bash
configs/infer_unet3d_benchmark.py
```

Kept infer config families:

- `configs/infer_unet3d.py`: old baseline infer profile
- `configs/infer_unet3d_benchmark.py`: current benchmark profile (`safe_unit_scale` + gaussian SW + in-plane TTA)
- `configs/infer_unet3d_reference_transforms.py`: special profile for the old reference-transforms model family

Val benchmark snapshots (`inputs/splits/val_case_days.csv`, 58 case_days) under the unified benchmark profile:

| Recipe | Weights | TTA | Mean Dice | Large bowel | Small bowel | Stomach | Example command |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Single | `20260319-173558_Unet3D_ref_loss_diceBCE_safe_aug/best.pt` | No | **0.7879** | 0.7955 | 0.6958 | 0.8725 | `python scripts/infer_unet3d.py --config configs/infer_unet3d_benchmark.py --weights outputs/20260319-173558_Unet3D_ref_loss_diceBCE_safe_aug/best.pt --disable_tta --output_dir outputs/benchmark_infer_single` |
| Single | `20260318-114555_Unet3D_ref_loss_diceBCE_epoch_150/best.pt` | in-plane flips `[[1], [2], [1, 2]]` | **0.7888** | 0.7937 | 0.7006 | 0.8722 | `python scripts/infer_unet3d.py --config configs/infer_unet3d_benchmark.py --weights outputs/20260318-114555_Unet3D_ref_loss_diceBCE_epoch_150/best.pt --output_dir outputs/benchmark_infer_single_tta` |
| Dual ensemble | `20260316-202538_Unet3D/best.pt` + `20260319-173558_Unet3D_ref_loss_diceBCE_safe_aug/best.pt` | in-plane flips `[[1], [2], [1, 2]]` | **0.7742** | 0.7851 | 0.6838 | 0.8538 | `python scripts/infer_unet3d.py --config configs/infer_unet3d_benchmark.py --weights outputs/20260316-202538_Unet3D/best.pt,outputs/20260319-173558_Unet3D_ref_loss_diceBCE_safe_aug/best.pt --output_dir outputs/benchmark_infer_dual_tta` |

Notes:
- The benchmark config is now generic; single vs ensemble and TTA on/off are CLI-level choices.
- `--weights` accepts either one checkpoint path or a comma-separated list for ensembles.
- `--disable_tta` forces `tta_flips=[]`; otherwise the benchmark profile uses in-plane flips by default.
- All benchmark runs use `roi_size=[96,224,224]`, `sw_batch_size=2`, `threshold=0.5`, `overlap=0.5`, `sw_mode="gaussian"`, and `intensity_norm="safe_unit_scale"`.
- Raw summaries are stored in:
  - `outputs/benchmark_infer_single/eval_summary.json`
  - `outputs/benchmark_infer_single_tta/eval_summary.json`
  - `outputs/benchmark_infer_dual_tta/eval_summary.json`

## Docs

- Training baseline details: `docs/TRAINING_BASELINE.md`
- nnUNet baseline details: `docs/NNUNET_BASELINE.md`
- MLflow tracking details: `docs/MLFLOW.md`
- Metrics protocol and evaluation bias controls: `docs/METRICS_PROTOCOL.md`
- Protocol benchmark script: `analysis/scripts/benchmark_metrics_protocol.py`

## nnUNet baseline

The repo now includes an `nnUNet v2` baseline that follows the official `DatasetXXX_Name` layout and uses the current recommended `ResEnc L` preset for `3d_fullres`.

Prepare the dataset:

```bash
python scripts/prepare_nnunet_dataset.py --config configs/train_nnunet_baseline.py
```

Run the full baseline with MLflow logging:

```bash
python scripts/train_nnunet_baseline.py --config configs/train_nnunet_baseline.py
```

Artifacts include:

- `nnUNet_raw/Dataset501_UWGI/dataset.json`
- `nnUNet_preprocessed/Dataset501_UWGI/splits_final.json`
- official `nnUNet` `validation/summary.json`
- repo-side auxiliary `eval_summary.json`, `eval_per_case.csv`, and `submit.csv`

Important caveat: this project's labels are originally multilabel, but the `nnUNet` baseline uses a lossy single-label collapse with fixed overwrite priority `stomach > small_bowel > large_bowel`. See `docs/NNUNET_BASELINE.md` for details.

## To update

Planned improvements (not included in this release). This section is meant as a roadmap and a checklist for future release.

### Metrics / evaluation correctness (no Kaggle submissions)

- Align local evaluation with the original competition metric (Kaggle submissions are no longer available for verification):
  - Current `scripts/infer_unet3d.py` reports Dice aggregated over the full 3D `case_day` volume.
  - Add a “Kaggle-equivalent” slice-level evaluator that computes Dice per 2D slice and aggregates with the same conventions
    (including how empty masks are handled).
  - Add correctness checks: RLE round-trip identity (`mask -> RLE -> mask`) and evaluator self-consistency tests.
- Extend reporting with additional metrics often used in medical segmentation:
  - surface distance metrics (e.g., HD95 / ASD)
  - per-class summary tables and stratified breakdowns (worst-case / site-like clusters)
  - threshold sweep / calibration curves (Dice vs threshold)

### Data / domain shift mitigation

- Intensity handling:
  - percentile clipping + robust normalization
  - histogram standardization / matching (site/vendor normalization)
  - optional dithering / quantization-aware preprocessing for vendor-specific artifacts
- Augmentations targeted at scanner variability (contrast/noise/blur) and geometric robustness.

### Class imbalance and loss design

- Foreground-aware sampling (minority-organ coverage):
  - ensure each batch/patch contains foreground for minority organs (or use class-balanced patch selection)
- Loss candidates to try (alone or in combinations):
  - Focal loss / Focal Dice
  - Tversky / Focal Tversky
  - boundary-aware losses (surface / level-set style) for thin structures
  - class-weighted Dice/CE and/or log-cosh Dice variants
- Consider objectives that emphasize minority-organ recall / PR-AUC while preserving overall Dice.

### Model and inference improvements


- Inference / TTA:
  - test-time augmentation (axis flips / rotations) with ensembling
  - TTA consistency checks (variance across augmentations) to flag uncertain regions
  - optional TTA-based consistency filtering to reduce spurious false-positive islands
  - post-processing: connected-component filtering / anatomy-aware constraints (class-specific size thresholds)
- Model candidates:
  - MONAI `SwinUNETR`, `DynUNet`, `SegResNet` (or lighter backbones for speed)
  - multi-scale / deep supervision variants
