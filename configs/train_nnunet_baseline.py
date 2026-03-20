from types import SimpleNamespace


cfg = SimpleNamespace(**{})

# data
cfg.data_root = "inputs"
cfg.train_csv = "inputs/train.csv"
cfg.train_ids = "inputs/splits/train_case_days.csv"
cfg.val_ids = "inputs/splits/val_case_days.csv"

# output
cfg.output_dir = ""
cfg.model_name = "nnUNet3D_fullres_baseline"

# nnUNet dataset + paths
cfg.dataset_id = 501
cfg.dataset_name = "UWGI"
cfg.nnunet_raw_dir = "nnUNet_raw"
cfg.nnunet_preprocessed_dir = "nnUNet_preprocessed"
cfg.nnunet_results_dir = "nnUNet_results"

# official baseline defaults
cfg.planner = "nnUNetPlannerResEncL"
cfg.plans = "nnUNetResEncUNetPlans"
cfg.trainer = "nnUNetTrainer"
cfg.configuration = "3d_fullres"
cfg.fold = 0
cfg.device = "cuda"
cfg.continue_training = False
cfg.verify_dataset_integrity = True

# workflow
cfg.run_prepare = True
cfg.run_preprocess = True
cfg.run_post_eval = True
cfg.overwrite = False
cfg.spacing_z = None

# label handling
cfg.label_collapse_mode = "priority_overwrite"
cfg.label_priority = ["large_bowel", "small_bowel", "stomach"]
cfg.threshold = 0.5

# tracking
cfg.use_mlflow = True
cfg.allow_dirty_git = True
cfg.mlflow_tracking_uri = "sqlite:///./mlflow.db"
cfg.mlflow_experiment = "uwgi/segmentation"
