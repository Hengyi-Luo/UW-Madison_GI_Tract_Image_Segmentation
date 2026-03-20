from types import SimpleNamespace


cfg = SimpleNamespace(**{})

# data
cfg.data_root = "inputs"
cfg.ids_csv = "inputs/splits/val_case_days.csv"
cfg.train_csv = "inputs/train.csv"

# weights / outputs
# Canonical benchmark inference profile:
# - current best single-model weights by default
# - override with --weights for a different checkpoint or an ensemble
cfg.weights = "outputs/20260319-173558_Unet3D_ref_loss_diceBCE_safe_aug/best.pt"
cfg.output_dir = "outputs/benchmark_infer"

# CSV + NIfTI outputs
cfg.out_csv_name = "submit.csv"
cfg.export_pred_nifti = False
cfg.pred_nifti_dir_name = "pred_nifti"

# evaluation
cfg.evaluate = True
cfg.eval_per_case_name = "eval_per_case.csv"
cfg.eval_summary_name = "eval_summary.json"

# sliding-window inference
cfg.roi_d = 96
cfg.roi_h = 224
cfg.roi_w = 224
cfg.sw_batch_size = 2
cfg.overlap = 0.5
cfg.sw_mode = "gaussian"
cfg.sw_sigma_scale = 0.125
cfg.threshold = 0.5
cfg.tta_flips = [[1], [2], [1, 2]]

# preprocessing
cfg.intensity_norm = "safe_unit_scale"
cfg.pad_to_roi = True

# runtime
cfg.num_workers = 0
cfg.device = ""

# checkpoint loading
cfg.unsafe_load = False
