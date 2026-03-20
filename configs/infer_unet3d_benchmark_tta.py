from types import SimpleNamespace


cfg = SimpleNamespace(**{})

# data
cfg.data_root = "inputs"
cfg.ids_csv = "inputs/splits/val_case_days.csv"
cfg.train_csv = "inputs/train.csv"

# weights / outputs
# Best-practice benchmark profile:
# - top self-trained baseline
# - top self-trained DiceBCE model
cfg.weights = [
    "outputs/20260316-202538_Unet3D/best.pt",
    "outputs/20260318-114555_Unet3D_ref_loss_diceBCE_epoch_150/best.pt",
]
cfg.output_dir = "outputs/benchmark_infer_tta"

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

# TTA best practice for this project:
# - use only in-plane flips
# - no depth flip
cfg.tta_flips = [[1], [2], [1, 2]]

# preprocessing
cfg.intensity_norm = "safe_unit_scale"
cfg.pad_to_roi = True

# runtime
cfg.num_workers = 0
cfg.device = ""

# checkpoint loading
cfg.unsafe_load = False
