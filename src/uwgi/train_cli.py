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
    p.add_argument("--epochs", type=int, default=defaults.get("epochs"))
    p.add_argument("--batch_size", type=int, default=defaults.get("batch_size"))
    p.add_argument("--lr", type=float, default=defaults.get("lr"))
    p.add_argument("--patch_d", type=int, default=defaults.get("patch_d"))
    p.add_argument("--patch_h", type=int, default=defaults.get("patch_h"))
    p.add_argument("--patch_w", type=int, default=defaults.get("patch_w"))
    p.add_argument("--samples_per_volume", type=int, default=defaults.get("samples_per_volume"))
    p.add_argument("--cache_in_ram", action="store_true")
    p.add_argument("--num_workers", type=int, default=defaults.get("num_workers"))
    p.add_argument("--seed", type=int, default=defaults.get("seed"))
    p.add_argument("--val_ratio", type=float, default=defaults.get("val_ratio"))
    p.add_argument("--verbose_shape_fix", action="store_true")
    p.add_argument("--mixed_precision", type=str, default=defaults.get("mixed_precision"))
    p.add_argument("--use_tensorboard", action="store_true")
    p.add_argument("--tb_dir", type=str, default=defaults.get("tb_dir"))
    p.add_argument("--log_steps", type=int, default=defaults.get("log_steps"))
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

    cfg = TrainCfg(
        data_root=args.data_root or "./input/uw-madison-gi-tract-image-segmentation",
        out=args.out or "./outputs/train_run/best.pt",
        run_dir=args.run_dir or "",
        epochs=args.epochs or 30,
        batch_size=args.batch_size or 1,
        lr=args.lr or 1e-4,
        patch_d=args.patch_d or 80,
        patch_h=args.patch_h or 224,
        patch_w=args.patch_w or 224,
        samples_per_volume=args.samples_per_volume or 12,
        cache_in_ram=args.cache_in_ram,
        num_workers=args.num_workers or 4,
        seed=args.seed or 42,
        val_ratio=args.val_ratio or 0.2,
        verbose_shape_fix=args.verbose_shape_fix,
        mixed_precision=args.mixed_precision or "no",
        use_tensorboard=args.use_tensorboard,
        tb_dir=args.tb_dir or "./outputs/train_run/tb",
        log_steps=args.log_steps or 50,
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
