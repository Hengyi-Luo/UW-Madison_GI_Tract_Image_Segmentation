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
- `configs/`: tracked YAML configs (release keeps only `train_unet3d.yaml` and `infer_unet3d.yaml`)
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

## Docs

- Training baseline details: `docs/TRAINING_BASELINE.md`
- MLflow tracking details: `docs/MLFLOW.md`

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

