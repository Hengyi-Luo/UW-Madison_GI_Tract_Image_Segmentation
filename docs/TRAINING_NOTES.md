# Training Notes

## Summary
- Current best training run: `20260319-173558_Unet3D_ref_loss_diceBCE_safe_aug`
- Best training metric: `0.78635`
- Main bottleneck class: `small_bowel`

## Key Runs
| Run | Change | Best Metric | Notes |
| --- | --- | ---: | --- |
| `20260316-202538_Unet3D` | Baseline | `0.78042` | Strong baseline with minimal transforms. |
| `20260318-114555_Unet3D_ref_loss_diceBCE_epoch_150` | `DiceBCE` loss + strong aug | `0.78254` | Small but real gain over baseline. |
| `20260316-205811_Unet3D_ref_aug` | Strong augmentation | `0.76539` | Worse than baseline. |
| `20260318-151445_Unet3D_ref_aug` | Strong augmentation | `0.76358` | Worse than baseline. |
| `20260319-173558_Unet3D_ref_loss_diceBCE_safe_aug` | `DiceBCE` + safer aug | `0.78635` | Best run so far. |

## What Helped
- `DiceBCE` improved over the baseline.
- Safer augmentation improved over both the baseline and the earlier strong-augmentation runs.
- Best current recipe:
  - `DiceBCE`
  - `safe_unit_scale`
  - H/W flips
  - light in-plane affine augmentation
  - no grid distortion
  - no coarse dropout

## What Did Not Help
- Strong augmentation with `RandGridDistortiond` and `RandCoarseDropoutd`
- Changing too many training variables at once

## Practical Notes
- Current stable training setup fits the 16GB GPU budget:
  - patch: `96x224x224`
  - train batch size: `8`
  - validation SW batch size: `4`
- Typical good runs use about `14GB` reserved VRAM.
- Final model selection should use formal inference benchmark results, not only training-time MLflow metrics.

## Next Focus
- Optimize `small_bowel` performance first.
- Keep the training backbone stable and only change one variable per experiment.
