import argparse
import sys

from .train_3d_monai import TrainCfg, train


def _load_yaml(path: str):
    try:
        import yaml
    except Exception:
        print("PyYAML not installed. Please: pip install pyyaml", file=sys.stderr)
        raise
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _build_parser(defaults):
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--data_root", type=str, default=defaults.get("data_root"))
    p.add_argument("--out", type=str, default=defaults.get("out"))
    p.add_argument("--run_dir", type=str, default=defaults.get("run_dir"))
    p.add_argument("--resume_from", type=str, default=defaults.get("resume_from"))
    p.add_argument("--save_last_every", type=int, default=defaults.get("save_last_every"))
    p.add_argument("--epochs", type=int, default=defaults.get("epochs"))
    p.add_argument("--batch_size", type=int, default=defaults.get("batch_size"))
    p.add_argument("--lr", type=float, default=defaults.get("lr"))
    p.add_argument("--patch_d", type=int, default=defaults.get("patch_d"))
    p.add_argument("--patch_h", type=int, default=defaults.get("patch_h"))
    p.add_argument("--patch_w", type=int, default=defaults.get("patch_w"))
    p.add_argument("--samples_per_volume", type=int, default=defaults.get("samples_per_volume"))
    p.add_argument("--train_fg_ratio", type=float, default=defaults.get("train_fg_ratio"))
    p.add_argument("--train_fg_min_voxels", type=int, default=defaults.get("train_fg_min_voxels"))
    p.add_argument("--train_fg_jitter", type=int, default=defaults.get("train_fg_jitter"))
    p.add_argument("--val_sw_every", type=int, default=defaults.get("val_sw_every"))
    p.add_argument("--val_sw_patch_d", type=int, default=defaults.get("val_sw_patch_d"))
    p.add_argument("--val_sw_patch_h", type=int, default=defaults.get("val_sw_patch_h"))
    p.add_argument("--val_sw_patch_w", type=int, default=defaults.get("val_sw_patch_w"))
    p.add_argument("--val_sw_batch_size", type=int, default=defaults.get("val_sw_batch_size"))
    p.add_argument("--val_sw_overlap", type=float, default=defaults.get("val_sw_overlap"))
    p.add_argument("--best_metric", type=str, default=defaults.get("best_metric"))
    p.add_argument("--val_full_spatial_order", type=str, default=defaults.get("val_full_spatial_order"))
    p.add_argument("--val_full_normalize", type=str, default=defaults.get("val_full_normalize"))
    p.add_argument(
        "--val_full_tta",
        dest="val_full_tta",
        action="store_true",
        default=bool(defaults.get("val_full_tta", False)),
        help="Enable flip TTA for full-volume kaggle_score eval (default: False).",
    )
    p.add_argument(
        "--no_val_full_tta",
        dest="val_full_tta",
        action="store_false",
        help="Disable flip TTA for full-volume kaggle_score eval.",
    )
    p.add_argument("--val_full_threshold", type=float, default=defaults.get("val_full_threshold"))
    p.add_argument("--val_full_hausdorff_empty_score", type=float, default=defaults.get("val_full_hausdorff_empty_score"))
    p.add_argument("--val_vis_every", type=int, default=defaults.get("val_vis_every"))
    p.add_argument("--val_vis_n", type=int, default=defaults.get("val_vis_n"))
    p.add_argument("--val_vis_threshold", type=float, default=defaults.get("val_vis_threshold"))
    p.add_argument("--val_vis_samples", type=str, default=defaults.get("val_vis_samples"))
    p.add_argument("--model", type=str, default=defaults.get("model"))
    p.add_argument("--feature_size", type=int, default=defaults.get("feature_size"))
    p.add_argument("--use_checkpoint", action="store_true", default=bool(defaults.get("use_checkpoint", False)))
    p.add_argument("--init_from", type=str, default=defaults.get("init_from"))
    p.add_argument("--cache_in_ram", action="store_true")
    p.add_argument("--num_workers", type=int, default=defaults.get("num_workers"))
    p.add_argument("--seed", type=int, default=defaults.get("seed"))
    p.add_argument("--val_ratio", type=float, default=defaults.get("val_ratio"))
    p.add_argument("--train_ids", type=str, default=defaults.get("train_ids"))
    p.add_argument("--val_ids", type=str, default=defaults.get("val_ids"))
    p.add_argument("--ids_column", type=str, default=defaults.get("ids_column"))
    p.add_argument("--verbose_shape_fix", action="store_true")
    p.add_argument("--mixed_precision", type=str, default=defaults.get("mixed_precision"))
    p.add_argument("--use_tensorboard", action="store_true")
    p.add_argument("--tb_dir", type=str, default=defaults.get("tb_dir"))
    p.add_argument("--log_steps", type=int, default=defaults.get("log_steps"))
    p.add_argument("--use_mlflow", action="store_true", default=bool(defaults.get("use_mlflow", True)))
    p.add_argument("--mlflow_tracking_uri", type=str, default=defaults.get("mlflow_tracking_uri"))
    p.add_argument("--mlflow_experiment", type=str, default=defaults.get("mlflow_experiment"))
    p.add_argument("--mlflow_run_name", type=str, default=defaults.get("mlflow_run_name"))
    p.add_argument("--mlflow_run_id", type=str, default=defaults.get("mlflow_run_id"))
    p.add_argument("--debug", action="store_true", default=bool(defaults.get("debug", False)))
    return p


def main():
    defaults = {}
    if "--config" in sys.argv:
        idx = sys.argv.index("--config")
        if idx + 1 < len(sys.argv):
            defaults = _load_yaml(sys.argv[idx + 1])
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        defaults = _load_yaml(sys.argv[1])

    p = _build_parser(defaults)
    args = p.parse_args()

    # Allow "epochs: null" (or omitted) in resume configs to mean:
    #   - if resuming: use total epochs from checkpoint cfg
    #   - otherwise: default to 30
    if args.epochs is None:
        epochs = None if args.resume_from else 30
    else:
        epochs = args.epochs

    cfg = TrainCfg(
        data_root=args.data_root or "./input/uw-madison-gi-tract-image-segmentation",
        out=args.out or "./outputs/train_run/best.pt",
        run_dir=args.run_dir or "",
        resume_from=args.resume_from or "",
        save_last_every=int(args.save_last_every or 1),
        epochs=epochs,
        batch_size=args.batch_size or 1,
        lr=args.lr or 1e-4,
        patch_d=args.patch_d or 96,
        patch_h=args.patch_h or 224,
        patch_w=args.patch_w or 224,
        samples_per_volume=args.samples_per_volume or 12,
        train_fg_ratio=args.train_fg_ratio if args.train_fg_ratio is not None else 0.0,
        train_fg_min_voxels=args.train_fg_min_voxels if args.train_fg_min_voxels is not None else 1,
        train_fg_jitter=args.train_fg_jitter if args.train_fg_jitter is not None else 0,
        val_sw_every=int(args.val_sw_every or 0),
        val_sw_patch_d=args.val_sw_patch_d,
        val_sw_patch_h=args.val_sw_patch_h,
        val_sw_patch_w=args.val_sw_patch_w,
        val_sw_batch_size=int(args.val_sw_batch_size or 1),
        val_sw_overlap=float(args.val_sw_overlap or 0.25),
        best_metric=args.best_metric or "patch",
        val_full_spatial_order=(args.val_full_spatial_order or "DHW"),
        val_full_normalize=(args.val_full_normalize or "none"),
        val_full_tta=bool(args.val_full_tta),
        val_full_threshold=float(args.val_full_threshold or 0.5),
        val_full_hausdorff_empty_score=float(args.val_full_hausdorff_empty_score or 0.0),
        val_vis_every=int(args.val_vis_every or 0),
        val_vis_n=int(args.val_vis_n or 16),
        val_vis_threshold=float(args.val_vis_threshold or 0.5),
        val_vis_samples=args.val_vis_samples
        or "./input/uw-madison-gi-tract-image-segmentation/splits/val_vis_samples.json",
        model=args.model or "swin_unetr",
        feature_size=args.feature_size or 48,
        use_checkpoint=bool(args.use_checkpoint),
        init_from=args.init_from or "",
        cache_in_ram=args.cache_in_ram,
        num_workers=args.num_workers or 4,
        seed=args.seed or 42,
        val_ratio=args.val_ratio or 0.2,
        train_ids=args.train_ids or "",
        val_ids=args.val_ids or "",
        ids_column=args.ids_column or "",
        verbose_shape_fix=args.verbose_shape_fix,
        mixed_precision=args.mixed_precision or "no",
        use_tensorboard=args.use_tensorboard,
        tb_dir=args.tb_dir or "./outputs/train_run/tb",
        log_steps=args.log_steps or 50,
        use_mlflow=args.use_mlflow,
        mlflow_tracking_uri=args.mlflow_tracking_uri or "",
        mlflow_experiment=args.mlflow_experiment or "uwgi",
        mlflow_run_name=args.mlflow_run_name or "",
        mlflow_run_id=args.mlflow_run_id or "",
        debug=args.debug,
    )

    if args.debug:
        if cfg.run_dir:
            cfg.run_dir = f"{cfg.run_dir}_debug"
        else:
            cfg.out = "./outputs/train_run_debug/best.pt"
            cfg.tb_dir = "./outputs/train_run_debug/tb"
        cfg.epochs = 1
        cfg.samples_per_volume = 1
        cfg.batch_size = 1
        cfg.num_workers = 0
        cfg.log_steps = 1
        cfg.mixed_precision = "no"

    train(cfg)


if __name__ == "__main__":
    main()
