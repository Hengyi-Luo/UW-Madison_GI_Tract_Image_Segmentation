from types import SimpleNamespace


cfg = SimpleNamespace(**{})

# data
cfg.data_root = "inputs"
cfg.ids_csv = "inputs/splits/val_case_days.csv"
cfg.train_csv = "inputs/train.csv"

# weights / outputs
cfg.weights = "outputs/20260316-132707_Unet3D_ref_aug_epoch_150/best.pt"
cfg.output_dir = ""

# CSV + NIfTI outputs
cfg.out_csv_name = "submit.csv"
cfg.export_pred_nifti = True
cfg.pred_nifti_dir_name = "pred_nifti"

# evaluation
cfg.evaluate = True
cfg.eval_per_case_name = "eval_per_case.csv"
cfg.eval_summary_name = "eval_summary.json"

# sliding-window inference
cfg.roi_d = 80
cfg.roi_h = 160
cfg.roi_w = 160
cfg.sw_batch_size = 4
cfg.overlap = 0.25
cfg.threshold = 0.5
cfg.intensity_norm = "safe_unit_scale"
cfg.pad_to_roi = False

# runtime
cfg.num_workers = 0
cfg.device = ""

# checkpoint loading
cfg.unsafe_load = False
