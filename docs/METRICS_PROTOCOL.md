# Metrics Protocol For 3D Multi-Organ Segmentation

This document defines the evaluation protocol for this repository's 3D GI tract segmentation task. The goal is not to maximize a single score, but to produce metrics that remain clinically interpretable and resistant to obvious evaluation bias.

The task is multi-label 3D segmentation on `case_day` volumes for:
- `large_bowel`
- `small_bowel`
- `stomach`

The protocol below is the recommended default for model comparison, ablation studies, and benchmark reporting.

## Principles

The evaluation must answer four separate questions:

1. Did the model detect that the organ is present?
2. If the organ is present, how well does the predicted extent overlap with GT?
3. If the organ is present, how well does the predicted boundary match GT?
4. If the organ is absent, how often does the model hallucinate the organ anyway?

No single metric answers all four. Dice alone is not enough. HD95 alone is not enough. Precision and recall alone are not enough.

## Recommended Metric Set

Use the following metric families together.

### Primary metrics

- `Per-organ Dice` on GT-positive cases only
- `Per-organ HD95` on cases where both GT and prediction are non-empty
- `Per-organ Recall` on GT-positive cases
- `Per-organ Precision` on prediction-positive cases

### Required safety / bias-control metrics

- `Per-organ GT-positive case count`
- `Per-organ GT-negative case count`
- `Per-organ prediction-positive case count`
- `Per-organ missed-organ case rate`
- `Per-organ spurious-organ case rate`
- `Per-organ HD95 valid count`
- `Per-organ HD95 invalid count`

### Strongly recommended auxiliary metrics

- `Per-organ Relative Volume Error (RVE)` on GT-positive cases
- `Per-organ empty-case false positive volume`
- `Per-organ case-level presence sensitivity`
- `Per-organ case-level absence specificity`

## Metric Definitions

All metrics are computed on the full 3D volume, not slice-by-slice.

Let:
- `P_c` be the predicted binary mask for organ `c` in one `case_day`
- `G_c` be the GT binary mask for organ `c`
- `|.|` be the number of foreground voxels

### 1. Dice

For valid cases:

`Dice(P, G) = 2 * |P ∩ G| / (|P| + |G|)`

Recommended validity rule:
- valid if `|G| > 0`
- invalid if `|G| == 0`

Interpretation:
- Dice should answer overlap quality when the organ truly exists.
- If GT is empty, Dice becomes a poor proxy because it can either inflate the score or hide false positives depending on convention.

### 2. Recall

`Recall(P, G) = TP / (TP + FN)`

Recommended validity rule:
- valid if `|G| > 0`
- invalid if `|G| == 0`

Interpretation:
- Recall answers how much of the true organ was found.
- If GT is absent, recall is undefined and should not be averaged.

### 3. Precision

`Precision(P, G) = TP / (TP + FP)`

Recommended validity rule:
- valid if `|P| > 0`
- invalid if `|P| == 0`

Interpretation:
- Precision answers how much of the predicted organ is actually correct.
- If prediction is empty, precision is undefined and should not be averaged.

### 4. HD95

HD95 is the 95th percentile Hausdorff distance between prediction and GT boundaries.

Recommended validity rule:
- valid if `|G| > 0` and `|P| > 0`
- invalid otherwise

Interpretation:
- HD95 is a boundary accuracy metric.
- If either side is empty, the boundary distance is not clinically interpretable as a normal finite surface mismatch.

### 5. Relative Volume Error

`RVE(P, G) = (|P| - |G|) / |G|`

Recommended validity rule:
- valid if `|G| > 0`
- invalid if `|G| == 0`

Interpretation:
- RVE shows whether the model systematically over-segments or under-segments.
- Dice can look acceptable even when volume bias is large.

## Empty-Case Policy

This is the most important part of the protocol.

### Dice

Recommended rule:
- `GT empty` -> do not include that case in Dice averaging

Reason:
- counting empty-empty as `1` inflates performance
- counting GT-empty and pred-nonempty as `0` mixes detection failure with overlap failure
- the cleaner approach is to move GT-empty behavior into separate presence / false-positive metrics

### Recall

Recommended rule:
- `GT empty` -> undefined -> ignore

Reason:
- recall only makes sense when the organ truly exists

### Precision

Recommended rule:
- `Prediction empty` -> undefined -> ignore

Reason:
- precision answers "of what I predicted, how much is correct?"
- if nothing is predicted, that question is undefined

### HD95

Recommended rule:
- `GT empty` and `Prediction empty` -> invalid -> ignore
- `GT empty` and `Prediction non-empty` -> invalid for HD95; count this as a spurious-organ case
- `GT non-empty` and `Prediction empty` -> invalid for HD95; count this as a missed-organ case

Reason:
- boundary distance is undefined when one or both boundaries do not exist
- forcing `0` or `inf` into the mean creates unstable and misleading summary statistics

## Required Aggregation Method

Do not aggregate voxels globally over the full dataset. That creates severe size bias.

### Step 1: per-case, per-organ

For each `case_day` and each organ:
- compute organ presence in GT
- compute organ presence in prediction
- compute valid metrics using the rules above
- record invalidity reason if not valid

### Step 2: per-organ dataset summary

For each organ:
- average Dice across valid cases only
- average Recall across valid cases only
- average Precision across valid cases only
- average HD95 across valid cases only
- average RVE across valid cases only
- report valid counts and invalid counts
- report missed-organ case rate
- report spurious-organ case rate

### Step 3: overall summary

Use macro average across organs:

- `macro_dice = mean(dice_large_bowel, dice_small_bowel, dice_stomach)`
- same for Recall, Precision, HD95, RVE

Do not use a voxel-weighted global average as the main headline metric.

Reason:
- large organs and long volumes would dominate the score
- smaller organs would be hidden

## Benchmark Report Format

Every model comparison should include the following table.

### Table A: per-organ summary

For each organ, report:
- `dice_mean_valid`
- `dice_valid_n`
- `recall_mean_valid`
- `recall_valid_n`
- `precision_mean_valid`
- `precision_valid_n`
- `hd95_mean_valid`
- `hd95_valid_n`
- `rve_mean_valid`
- `gt_positive_n`
- `gt_negative_n`
- `pred_positive_n`
- `missed_organ_rate`
- `spurious_organ_rate`

### Table B: overall summary

Report:
- `macro_dice`
- `macro_recall`
- `macro_precision`
- `macro_hd95`
- `macro_rve`

### Table C: failure summary

Report:
- worst 10 cases by Dice
- worst 10 cases by HD95
- worst 10 GT-negative hallucination cases by predicted volume
- worst 10 GT-positive missed-organ cases

## Severe Bias Risks

These are the main evaluation biases that can break interpretability.

### 1. Empty-case inflation bias

Problem:
- If empty-empty Dice is counted as `1`, the score can become artificially high.

Why it is severe:
- organs that are absent in many cases appear easier than they are
- models that simply predict nothing get rewarded

Mitigation:
- ignore GT-empty cases for Dice
- separately report spurious-organ rates on GT-negative cases

### 2. Hidden false-positive bias

Problem:
- If GT-empty cases are ignored for Dice and nothing else is reported, hallucinated organs disappear from the main evaluation.

Why it is severe:
- the model may look safe in Dice while clinically producing false structures

Mitigation:
- report case-level absence specificity
- report GT-negative false positive case rate
- report GT-negative false positive volume

### 3. Organ-size dominance bias

Problem:
- Large organs dominate voxel-level or micro-averaged metrics.

Why it is severe:
- performance on small or difficult organs is hidden

Mitigation:
- report per-organ metrics first
- use macro average over organs as headline summary

### 4. Long-volume dominance bias

Problem:
- Cases with more slices or more foreground voxels dominate dataset-level aggregation.

Why it is severe:
- the evaluation becomes a weighted study of a few large volumes

Mitigation:
- compute metrics per case first
- then average over cases

### 5. Overlap-only bias

Problem:
- Dice can look acceptable even when boundaries are clinically poor.

Why it is severe:
- small boundary shifts in thin structures matter clinically but can be hidden by overlap metrics

Mitigation:
- always pair Dice with a boundary metric such as HD95

### 6. Boundary-censoring bias

Problem:
- If invalid HD95 cases are silently dropped without counts, the mean HD95 looks better than it should.

Why it is severe:
- missed organs and hallucinated organs disappear from boundary reporting

Mitigation:
- always report `hd95_valid_n`
- always report missed-organ and spurious-organ rates

### 7. Spacing bias

Problem:
- HD95 in voxel units is not comparable across anisotropic data if physical spacing differs.

Why it is severe:
- the same boundary error can look very different across scans

Mitigation:
- use true physical spacing whenever available
- if spacing is incomplete, explicitly state that HD95 is voxel-space only

### 8. Threshold-selection bias

Problem:
- tuning the threshold on the evaluation split inflates the final score.

Why it is severe:
- the reported benchmark becomes optimistic and non-reproducible

Mitigation:
- tune threshold on a separate development split
- freeze the threshold before benchmark reporting

### 9. Slice-vs-volume mismatch bias

Problem:
- 2D slice-wise metrics and 3D volume-wise metrics can rank models differently.

Why it is severe:
- the benchmark target changes depending on implementation details

Mitigation:
- choose one official protocol and state it explicitly
- for this repository, use 3D `case_day` volume evaluation as the default protocol

### 10. Orientation / decoding bias

Problem:
- RLE decode bugs, axis ordering mistakes, and slice sorting errors can produce seemingly plausible but wrong metrics.

Why it is severe:
- the model can appear good or bad for purely technical reasons

Mitigation:
- require mask -> RLE -> mask round-trip checks
- verify slice ordering and channel ordering in the benchmark

### 11. Post-processing bias

Problem:
- connected-component filtering or size thresholding can improve one metric while damaging another.

Why it is severe:
- improvements may be post-processing artifacts rather than actual model quality

Mitigation:
- benchmark the full inference pipeline end-to-end
- freeze post-processing when comparing models

### 12. Prevalence bias

Problem:
- organs with lower prevalence or more empty cases can look artificially good or bad depending on the empty-case rule.

Why it is severe:
- cross-organ comparison becomes misleading

Mitigation:
- always report GT-positive and GT-negative counts per organ

## Recommended Default Protocol For This Repository

For this specific 3D multi-organ task, the recommended default is:

### Main comparison metrics

- `Per-organ Dice` on GT-positive cases only
- `Per-organ HD95` on cases with both GT and prediction non-empty
- `Per-organ Recall` on GT-positive cases
- `Per-organ Precision` on prediction-positive cases
- `Macro average across organs`

### Mandatory bias-control metrics

- `GT-positive count`
- `GT-negative count`
- `missed-organ rate`
- `spurious-organ rate`
- `HD95 valid count`

### Practical interpretation

Use the metrics together:
- low Dice + low Recall -> under-segmentation / misses
- low Dice + low Precision -> over-segmentation / spurious regions
- acceptable Dice + high HD95 -> overlap is okay but boundary is clinically poor
- good Dice + high spurious-organ rate -> empty-case bias is hiding hallucinations

## Minimal Pseudocode

For each `case_day`:

1. reconstruct full 3D prediction and GT for each organ
2. compute:
   - `gt_present = |G| > 0`
   - `pred_present = |P| > 0`
3. if `gt_present`:
   - compute Dice
   - compute Recall
   - compute RVE
4. if `pred_present`:
   - compute Precision
5. if `gt_present and pred_present`:
   - compute HD95
6. if `gt_present and not pred_present`:
   - mark missed-organ case
7. if `not gt_present and pred_present`:
   - mark spurious-organ case
8. aggregate per organ over valid cases only
9. macro-average over organs
10. report valid counts next to every metric

## What Should Not Be The Only Headline Metric

Do not use these alone:
- global voxel-wise accuracy
- micro-averaged Dice over all voxels
- Dice without empty-case policy
- HD95 without valid counts
- Precision / Recall without validity rules

## Bottom Line

For this task, the benchmark should not be "one Dice number."

The benchmark should be:
- macro per-organ Dice
- macro per-organ HD95
- macro per-organ Recall
- macro per-organ Precision
- plus organ presence / absence failure rates and valid counts

That is the smallest metric set that remains clinically interpretable and resistant to obvious evaluation bias.
