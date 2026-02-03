from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.utils import set_determinism

from .checkpointing import (
    apply_cfg_from_checkpoint,
    checkpoint_paths,
    load_init_weights,
    load_resume_checkpoint,
    move_optimizer_state_to_device,
)
from .constants import CLASSES
from .data_utils import build_case_day_slices, build_rle_index, load_case_days
from .datasets import UWGI3DPatchDataset
from .datasets import UWGI3DFullVolumeDataset
from .engine import train_one_epoch, validate_dice, validate_sliding_window_dice
from .experiment import git_info, resolve_run_naming
from .metrics import dice_per_class_2d_kaggle, hausdorff_score_3d_scipy_kaggle
from .models import build_model
from .splits import split_case_days


def _load_val_vis_samples(path: str) -> list[dict]:
    """
    Load a fixed set of val visualization samples from JSON.

    Supported formats:
      - list[dict] (legacy)
      - {"samples": list[dict], ...} (recommended)

    Each sample must contain:
      - case_day: str
      - slice_pos: int  (index into case_day_slices[case_day])
    Optional:
      - slice_idx: int  (for nicer TB tags)
    """
    if not path:
        return []
    if not os.path.exists(path):
        print(f"[WARN] val_vis_samples not found: {path}")
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read val_vis_samples: {path} ({e})")
        return []

    if isinstance(obj, dict) and isinstance(obj.get("samples"), list):
        samples = obj["samples"]
    elif isinstance(obj, list):
        samples = obj
    else:
        print(f"[WARN] Unrecognized val_vis_samples JSON shape: {type(obj)}")
        return []

    out: list[dict] = []
    for s in samples:
        if not isinstance(s, dict):
            continue
        if "case_day" not in s or "slice_pos" not in s:
            continue
        out.append(s)
    return out


def _default_val_vis_samples(
    val_days: list[str],
    case_day_slices: dict[str, list[str]],
    *,
    n: int = 16,
) -> list[dict]:
    """
    Fallback val samples when `val_vis_samples` JSON is missing.
    Picks the first N case_days and uses the middle slice.
    """
    out: list[dict] = []
    for cd in val_days:
        paths = case_day_slices.get(cd) or []
        if not paths:
            continue
        slice_pos = int(len(paths) // 2)
        out.append({"case_day": str(cd), "slice_pos": int(slice_pos), "slice_idx": -1})
        if len(out) >= int(n):
            break
    return out


@torch.no_grad()
def _infer_volume_pred_mask(
    x_bcdhw: torch.Tensor,
    *,
    model: torch.nn.Module,
    device: torch.device,
    roi_size: tuple[int, int, int],
    sw_batch_size: int,
    overlap: float,
    threshold: float,
    spatial_order: str,
    normalize: str,
    tta: bool,
) -> torch.Tensor:
    """
    Returns pred mask (C,D,H,W) uint8 in {0,1}.

    x_bcdhw: (B,1,D,H,W) float32 in [0,1]
    """
    from monai.inferers import sliding_window_inference

    x_in = x_bcdhw
    spatial_order = (spatial_order or "DHW").upper()
    if spatial_order not in {"DHW", "WHD"}:
        raise ValueError(f"Unsupported spatial_order={spatial_order}. Supported: DHW, WHD")
    if spatial_order == "WHD":
        # (B,C,D,H,W) -> (B,C,W,H,D)
        x_in = x_in.permute(0, 1, 4, 3, 2)

    normalize = (normalize or "none").lower()
    if normalize not in {"none", "volume_max"}:
        raise ValueError(f"Unsupported normalize={normalize}. Supported: none, volume_max")
    if normalize == "volume_max":
        mx = x_in.amax(dim=(2, 3, 4), keepdim=True).clamp_min(1e-6)
        x_in = x_in / mx

    if spatial_order == "WHD":
        # flip W/H like reference kernel
        dims_list = [(), (2,), (3,), (2, 3)]
    else:
        # flip H/W
        dims_list = [(), (3,), (4,), (3, 4)]
    if not bool(tta):
        dims_list = [()]

    logits_sum = None
    # NOTE: this intentionally does NOT use accelerator.autocast() to better match
    # notebook-style validation (val_pipeline.ipynb) and avoid fp16 threshold edge flips.
    for dims in dims_list:
        x_f = torch.flip(x_in, dims=dims) if dims else x_in
        l = sliding_window_inference(
            x_f.to(device, non_blocking=True),
            roi_size,
            int(sw_batch_size),
            model,
            overlap=float(overlap),
        )
        if dims:
            l = torch.flip(l, dims=dims)
        logits_sum = l if logits_sum is None else (logits_sum + l)
    logits = logits_sum / float(len(dims_list))

    if spatial_order == "WHD":
        # (B,C,W,H,D) -> (B,C,D,H,W)
        logits = logits.permute(0, 1, 4, 3, 2)

    prob = torch.sigmoid(logits)
    pred = (prob > float(threshold)).to(torch.uint8)[0].cpu()  # (C,D,H,W)
    return pred


def _run_full_val_kaggle_and_save(
    *,
    model: torch.nn.Module,
    val_loader,
    accelerator: Accelerator,
    device: torch.device,
    epoch: int,
    out_dir: str,
    roi_size: tuple[int, int, int],
    sw_batch_size: int,
    overlap: float,
    threshold: float,
    spatial_order: str,
    normalize: str,
    tta: bool,
    case_day_slices: dict[str, list[str]],
    rle_index,
    vis_samples: list[dict],
    vis_n: int,
    hausdorff_empty_score: float = 0.0,
) -> dict:
    """
    Full-volume validation like `val_pipeline.ipynb`:
      - predict per case_day volume with SW (+ optional TTA/normalize/spatial order)
      - compute Kaggle-style per-slice Dice + 3D Hausdorff score
      - write eval_per_case.csv + eval_summary.json
      - save up to N visualization samples to disk
    """
    if not accelerator.is_main_process:
        return {}

    import numpy as np
    from torch.utils.data import DataLoader

    os.makedirs(out_dir, exist_ok=True)

    # In multi-process training, Accelerate may shard the prepared val_loader.
    # For a true "full" validation, rebuild a plain DataLoader over the underlying dataset on main.
    dl = val_loader
    try:
        if int(getattr(accelerator, "num_processes", 1) or 1) > 1:
            ds = getattr(val_loader, "dataset", None)
            if ds is not None:
                dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    except Exception:
        dl = val_loader

    samples = (vis_samples or [])[: int(vis_n)]
    sample_case_days = {str(s.get("case_day")) for s in samples if isinstance(s, dict) and s.get("case_day")}
    cache_vols: dict[str, dict] = {}

    per_case_rows: list[dict] = []

    dice_sum_all = 0.0
    dice_count_all = 0
    dice_sum_nonempty = 0.0
    dice_count_nonempty = 0
    empty_empty_pairs = 0
    gt_non_empty_total = 0

    hd_sum = 0.0
    hd_count = 0

    model.eval()

    for x, y, case_day in dl:
        name = case_day[0] if isinstance(case_day, (list, tuple)) else str(case_day)
        x = x.to(device, non_blocking=True)
        y_np = (y[0].cpu().numpy() > 0.5).astype(np.uint8)  # (C,D,H,W)

        pred_t = _infer_volume_pred_mask(
            x,
            model=model,
            device=device,
            roi_size=roi_size,
            sw_batch_size=int(sw_batch_size),
            overlap=float(overlap),
            threshold=float(threshold),
            spatial_order=str(spatial_order),
            normalize=str(normalize),
            tta=bool(tta),
        )
        pred_np = pred_t.numpy()  # (C,D,H,W) uint8

        c, d, _, _ = pred_np.shape

        # Dice: mean over (slice, class) pairs, empty-empty => 0
        case_dice_sum = 0.0
        case_dice_count = 0
        case_dice_sum_nonempty = 0.0
        case_dice_count_nonempty = 0
        case_empty_empty = 0
        case_gt_non_empty = 0

        for z in range(int(d)):
            pred2d = pred_np[:, z]
            gt2d = y_np[:, z]
            dice_c = dice_per_class_2d_kaggle(pred2d, gt2d)  # (C,)
            case_dice_sum += float(dice_c.sum())
            case_dice_count += int(dice_c.size)

            pred_sum = pred2d.reshape(int(c), -1).sum(axis=1)
            gt_sum = gt2d.reshape(int(c), -1).sum(axis=1)
            denom = pred_sum + gt_sum
            nonempty = denom > 0
            case_empty_empty += int((~nonempty).sum())
            case_gt_non_empty += int((gt_sum > 0).sum())
            if bool(nonempty.any()):
                case_dice_sum_nonempty += float(dice_c[nonempty].sum())
                case_dice_count_nonempty += int(nonempty.sum())

        case_dice = float(case_dice_sum / max(1, case_dice_count))
        case_dice_nonempty = (
            float(case_dice_sum_nonempty / max(1, case_dice_count_nonempty)) if case_dice_count_nonempty > 0 else 0.0
        )

        dice_sum_all += float(case_dice_sum)
        dice_count_all += int(case_dice_count)
        dice_sum_nonempty += float(case_dice_sum_nonempty)
        dice_count_nonempty += int(case_dice_count_nonempty)
        empty_empty_pairs += int(case_empty_empty)
        gt_non_empty_total += int(case_gt_non_empty)

        # Hausdorff (3D) score: mean over classes, empty-empty => empty_score (default=0)
        hd_by_cls, hd_mean = hausdorff_score_3d_scipy_kaggle(
            pred_np,
            y_np,
            empty_score=float(hausdorff_empty_score),
            return_per_class=True,
        )
        hd_sum += float(hd_by_cls.sum())
        hd_count += int(hd_by_cls.size)

        case_final_score = 0.4 * float(case_dice) + 0.6 * float(hd_mean)

        per_case_rows.append(
            {
                "case_day": str(name),
                "dice": float(case_dice),
                "dice_nonempty": float(case_dice_nonempty),
                "hausdorff_score": float(hd_mean),
                "final_score": float(case_final_score),
                "count_total": int(case_dice_count),
                "count_nonempty": int(case_dice_count_nonempty),
                "gt_non_empty": int(case_gt_non_empty),
                "num_slices": int(d),
                "num_classes": int(c),
            }
        )

        if name in sample_case_days:
            cache_vols[name] = {"pred": pred_np, "gt": y_np}

    mean_dice_all = float(dice_sum_all / max(1, dice_count_all))
    mean_dice_nonempty = float(dice_sum_nonempty / max(1, dice_count_nonempty)) if dice_count_nonempty > 0 else 0.0
    mean_hausdorff_score = float(hd_sum / max(1, hd_count)) if hd_count > 0 else 0.0
    final_score = 0.4 * float(mean_dice_all) + 0.6 * float(mean_hausdorff_score)

    per_case_rows.sort(key=lambda r: str(r.get("case_day", "")))
    out_csv = os.path.join(out_dir, "eval_per_case.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "case_day",
                "dice",
                "dice_nonempty",
                "hausdorff_score",
                "final_score",
                "count_total",
                "count_nonempty",
                "gt_non_empty",
                "num_slices",
                "num_classes",
            ],
        )
        w.writeheader()
        w.writerows(per_case_rows)

    out_json = os.path.join(out_dir, "eval_summary.json")

    def _csv_preview_md(path: str, *, sort_key: str = "final_score", n_each: int = 8) -> str:
        try:
            with open(path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except Exception as e:
            return f"Failed to read CSV: {path} ({e})"

        def _f(v: object) -> float:
            try:
                return float(v)  # type: ignore[arg-type]
            except Exception:
                return float("nan")

        if not rows:
            return f"(empty) {path}"

        rows_sorted = sorted(rows, key=lambda r: _f(r.get(sort_key)))
        worst = rows_sorted[: int(n_each)]
        best = rows_sorted[-int(n_each) :] if len(rows_sorted) > int(n_each) else []
        best = list(reversed(best))

        cols = ["case_day", "dice", "dice_nonempty", "hausdorff_score", "final_score"]

        def _md_table(title: str, part: list[dict]) -> str:
            if not part:
                return ""
            out = [f"### {title}", "", "| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
            for r in part:
                out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
            out.append("")
            return "\n".join(out)

        md = [
            f"CSV saved: `{path}`",
            "",
            _md_table("Worst cases (by final_score)", worst),
            _md_table("Best cases (by final_score)", best),
        ]
        return "\n".join([x for x in md if x])

    summary = {
        "timestamp": int(time.time()),
        "epoch": int(epoch),
        "metric": "kaggle_score",
        "score_weights": {"dice": 0.4, "hausdorff": 0.6},
        "mode": "sw",
        "roi_size": list(map(int, roi_size)),
        "spatial_order": str(spatial_order),
        "sw_batch_size": int(sw_batch_size),
        "overlap": float(overlap),
        "threshold": float(threshold),
        "normalize": str(normalize),
        "tta": bool(tta),
        "mean_dice": float(mean_dice_all),
        "mean_dice_all": float(mean_dice_all),
        "mean_dice_nonempty": float(mean_dice_nonempty),
        "count_nonempty": int(dice_count_nonempty),
        "empty_empty_pairs": int(empty_empty_pairs),
        "mean_hausdorff_score": float(mean_hausdorff_score),
        "final_score": float(final_score),
        "hausdorff_impl": "scipy_directed_hausdorff",
        "hausdorff_empty_score": float(hausdorff_empty_score),
        "count_total": int(dice_count_all),
        "gt_non_empty": int(gt_non_empty_total),
        "hd_score_count": int(hd_count),
        "num_case_days": int(len(per_case_rows)),
        "out_dir": str(out_dir),
        "out_per_case_csv": str(out_csv),
        "out_summary_json": str(out_json),
        "tb_csv_preview_md": _csv_preview_md(out_csv),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, indent=2, ensure_ascii=False))

    # Save sample visualizations (16 by default)
    if samples:
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"[WARN] Skipping sample saving (matplotlib not available): {e}")
            return summary

        try:
            from .viz import plot_case_day_slice_gt_pred, predict_case_day_volume

            sample_dir = os.path.join(out_dir, "samples")
            os.makedirs(sample_dir, exist_ok=True)

            for i, s in enumerate(samples):
                case_day = str(s.get("case_day"))
                slice_pos = int(s.get("slice_pos", 0))
                slice_idx = int(s.get("slice_idx", -1))

                vols = cache_vols.get(case_day)
                if vols is None:
                    # Fallback: compute just this case_day for visualization.
                    pred_vol, gt_vol, _ = predict_case_day_volume(
                        case_day,
                        case_day_slices=case_day_slices,
                        rle_index=rle_index,
                        model=model,
                        device=str(device),
                        mode="sw",
                        roi_size=roi_size,
                        sw_batch_size=int(sw_batch_size),
                        overlap=float(overlap),
                        threshold=float(threshold),
                        spatial_order=str(spatial_order),
                        normalize=str(normalize),
                        tta=bool(tta),
                    )
                else:
                    pred_vol = vols["pred"]
                    gt_vol = vols["gt"]

                fig_grid, fig_overlay, _, _, _ = plot_case_day_slice_gt_pred(
                    case_day,
                    slice_pos=slice_pos,
                    case_day_slices=case_day_slices,
                    rle_index=rle_index,
                    pred_vol=pred_vol,
                    gt_vol=gt_vol,
                )
                base = f"{i:02d}_{case_day}_slicepos_{slice_pos:03d}_sliceidx_{slice_idx:04d}"
                fig_grid.savefig(os.path.join(sample_dir, f"{base}__grid.png"), dpi=150, bbox_inches="tight")
                fig_overlay.savefig(os.path.join(sample_dir, f"{base}__overlay.png"), dpi=150, bbox_inches="tight")
                plt.close(fig_grid)
                plt.close(fig_overlay)
        except Exception as e:
            print(f"[WARN] Sample saving failed: {e}")

    return summary


def _log_val_samples_to_tb(
    *,
    writer: SummaryWriter,
    epoch: int,
    samples: list[dict],
    case_day_slices: dict[str, list[str]],
    rle_index,
    model,
    roi_size: tuple[int, int, int],
    sw_batch_size: int,
    overlap: float,
    threshold: float,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] Skipping TB val sample plots (matplotlib not available): {e}")
        return

    from .viz import plot_case_day_slice_gt_pred

    model_for_viz = model

    for i, s in enumerate(samples):
        case_day = str(s["case_day"])
        slice_pos = int(s["slice_pos"])
        slice_idx = int(s.get("slice_idx", -1))
        tag_base = f"val_samples/{i:02d}_{case_day}_slice_{slice_idx:04d}"

        fig_grid, fig_overlay, _, _, _ = plot_case_day_slice_gt_pred(
            case_day,
            slice_pos=slice_pos,
            case_day_slices=case_day_slices,
            rle_index=rle_index,
            model=model_for_viz,
            mode="sw",
            roi_size=roi_size,
            sw_batch_size=int(sw_batch_size),
            overlap=float(overlap),
            threshold=float(threshold),
        )
        writer.add_figure(f"{tag_base}/grid", fig_grid, global_step=epoch, close=True)
        writer.add_figure(f"{tag_base}/overlay", fig_overlay, global_step=epoch, close=True)
        plt.close("all")


@dataclass
class TrainCfg:
    # Data
    data_root: str = "/home/jeremiah/github/UW-Madison_GI_Tract_Image_Segmentation/inputs"
    seed: int = 42
    val_ratio: float = 0.2
    train_ids: str = ""  # optional explicit train case_day list file (csv/txt/json)
    val_ids: str = ""  # optional explicit val case_day list file (csv/txt/json)
    ids_column: str = ""  # optional csv column name for train_ids/val_ids
    num_workers: int = 4
    cache_in_ram: bool = False

    # Patches
    patch_d: int = 96
    patch_h: int = 224
    patch_w: int = 224
    samples_per_volume: int = 12
    train_fg_ratio: float = 0.0
    train_fg_min_voxels: int = 1
    train_fg_jitter: int = 0

    # Validation
    val_sw_every: int = 0  # 0 = disabled (patch-only); otherwise run full-volume SW val every N epochs
    val_sw_patch_d: int | None = None
    val_sw_patch_h: int | None = None
    val_sw_patch_w: int | None = None
    val_sw_batch_size: int = 1
    val_sw_overlap: float = 0.25
    best_metric: str = "patch"  # "patch" | "sw"

    # Full-volume Kaggle-style score (val_pipeline.ipynb)
    # Triggered on the same schedule as `val_sw_every`.
    val_full_spatial_order: str = "DHW"
    val_full_normalize: str = "none"
    val_full_tta: bool = False
    val_full_threshold: float = 0.5
    val_full_hausdorff_empty_score: float = 0.0

    # TensorBoard val sample visualization (full-slice GT vs Pred)
    val_vis_every: int = 0  # 0 = disabled; otherwise log every N epochs
    val_vis_n: int = 16
    val_vis_threshold: float = 0.5
    val_vis_samples: str = "/home/jeremiah/github/UW-Madison_GI_Tract_Image_Segmentation/inputs/splits/val_vis_samples.json"

    # Model
    model: str = "swin_unetr"  # unet | swin_unetr
    feature_size: int = 48
    use_checkpoint: bool = False
    init_from: str = ""  # optional pretrained init (shape-matched)

    # Optimization
    epochs: int | None = 30
    batch_size: int = 1
    lr: float = 1e-4

    # Output / resume
    out: str = "./checkpoints/best.pt"
    run_dir: str = ""
    resume_from: str = ""  # path to .../last.pt
    save_last_every: int = 1

    # Runtime
    mixed_precision: str = "no"  # "no", "fp16", "bf16"
    verbose_shape_fix: bool = False

    # Logging
    use_tensorboard: bool = False
    tb_dir: str = "./outputs/train_run/tb"
    log_steps: int = 50
    use_mlflow: bool = True
    mlflow_tracking_uri: str = ""
    mlflow_experiment: str = "uwgi"
    mlflow_run_name: str = ""
    mlflow_run_id: str = ""
    debug: bool = False


def train(cfg: TrainCfg) -> None:
    set_determinism(seed=cfg.seed)
    ckpt = load_resume_checkpoint(cfg.resume_from)

    if ckpt is not None and cfg.resume_from:
        apply_cfg_from_checkpoint(cfg, ckpt)
        resume_dir = os.path.dirname(cfg.resume_from)
        if (not cfg.run_dir) or ("{" in cfg.run_dir and "}" in cfg.run_dir):
            cfg.run_dir = resume_dir
        if not cfg.mlflow_run_id:
            cfg.mlflow_run_id = ckpt.get("mlflow_run_id", "") or ckpt.get("mlflow", {}).get("run_id", "")

    resolve_run_naming(cfg)
    if cfg.run_dir:
        cfg.out = os.path.join(cfg.run_dir, "best.pt")
        if cfg.use_tensorboard:
            cfg.tb_dir = os.path.join(cfg.run_dir, "tb")
    os.makedirs(os.path.dirname(cfg.out), exist_ok=True)

    train_dir = os.path.join(cfg.data_root, "train")
    train_csv = os.path.join(cfg.data_root, "train.csv")

    print("Building slice index...")
    case_day_slices = build_case_day_slices(train_dir)
    all_case_days = sorted(case_day_slices.keys())
    print(f"Found case_day volumes: {len(all_case_days)}")

    print("Building RLE index...")
    rle_index = build_rle_index(train_csv)

    if cfg.train_ids or cfg.val_ids:
        col = str(cfg.ids_column).strip() or None
        train_set = set(load_case_days(cfg.train_ids, column=col)) if cfg.train_ids else set()
        val_set = set(load_case_days(cfg.val_ids, column=col)) if cfg.val_ids else set()

        missing_train = sorted([cd for cd in train_set if cd not in case_day_slices])
        missing_val = sorted([cd for cd in val_set if cd not in case_day_slices])
        if missing_train:
            print(f"[WARN] train_ids missing from train/ directory: {len(missing_train)} (e.g. {missing_train[:5]})")
        if missing_val:
            print(f"[WARN] val_ids missing from train/ directory: {len(missing_val)} (e.g. {missing_val[:5]})")
        train_set = {cd for cd in train_set if cd in case_day_slices}
        val_set = {cd for cd in val_set if cd in case_day_slices}

        if cfg.train_ids and cfg.val_ids:
            overlap = sorted(train_set & val_set)
            if overlap:
                raise ValueError(f"train_ids and val_ids overlap (n={len(overlap)}), e.g. {overlap[:5]}")
            train_days = sorted(train_set)
            val_days = sorted(val_set)
        elif cfg.val_ids:
            val_days = [cd for cd in all_case_days if cd in val_set]
            train_days = [cd for cd in all_case_days if cd not in val_set]
        else:
            train_days = [cd for cd in all_case_days if cd in train_set]
            val_days = [cd for cd in all_case_days if cd not in train_set]

        if not train_days:
            raise ValueError("No train case_days after applying train_ids/val_ids filtering.")
        if not val_days:
            raise ValueError("No val case_days after applying train_ids/val_ids filtering.")
    else:
        train_days, val_days = split_case_days(all_case_days, val_ratio=cfg.val_ratio, seed=cfg.seed)
    print(f"Train volumes: {len(train_days)} | Val volumes: {len(val_days)}")

    split_dir = cfg.run_dir or os.path.dirname(cfg.out) or "."
    os.makedirs(split_dir, exist_ok=True)
    split_path = os.path.join(split_dir, "split_case_days.json")
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(
            {"seed": int(cfg.seed), "val_ratio": float(cfg.val_ratio), "train": train_days, "val": val_days},
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Saved train/val split: {split_path}")

    # Load a fixed set of val slices for TensorBoard visualization (no in-training sampling).
    val_vis_samples: list[dict] = _load_val_vis_samples(str(cfg.val_vis_samples))
    if val_vis_samples:
        val_vis_samples = val_vis_samples[: int(cfg.val_vis_n)]
        print(f"Loaded val samples: {len(val_vis_samples)} ({cfg.val_vis_samples})")
    else:
        # Fallback to deterministic defaults so we can still save 16 samples during full-volume val.
        val_vis_samples = _default_val_vis_samples(val_days, case_day_slices, n=int(cfg.val_vis_n))
        print(f"[WARN] No val samples loaded; using defaults: {len(val_vis_samples)}")

    patch_size = (int(cfg.patch_d), int(cfg.patch_h), int(cfg.patch_w))
    train_ds = UWGI3DPatchDataset(
        case_days=train_days,
        case_day_slices=case_day_slices,
        rle_index=rle_index,
        patch_size=patch_size,
        samples_per_volume=int(cfg.samples_per_volume),
        fg_ratio=float(cfg.train_fg_ratio),
        fg_min_voxels=int(cfg.train_fg_min_voxels),
        fg_jitter=int(cfg.train_fg_jitter),
        cache_in_ram=bool(cfg.cache_in_ram),
        seed=int(cfg.seed),
        is_train=True,
        verbose_shape_fix=bool(cfg.verbose_shape_fix),
    )
    val_ds = UWGI3DPatchDataset(
        case_days=val_days,
        case_day_slices=case_day_slices,
        rle_index=rle_index,
        patch_size=patch_size,
        samples_per_volume=max(1, int(cfg.samples_per_volume) // 2),
        fg_ratio=0.0,
        cache_in_ram=bool(cfg.cache_in_ram),
        seed=int(cfg.seed) + 999,
        is_train=False,
        verbose_shape_fix=False,
    )
    val_full_loader = None
    if int(cfg.val_sw_every) > 0:
        val_full_ds = UWGI3DFullVolumeDataset(
            case_days=val_days,
            case_day_slices=case_day_slices,
            rle_index=rle_index,
            cache_in_ram=False,
            verbose_shape_fix=False,
        )
        val_full_loader = DataLoader(
            val_full_ds,
            batch_size=1,
            shuffle=False,
            num_workers=int(cfg.num_workers),
            pin_memory=True,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=True,
    )

    accelerator = Accelerator(mixed_precision=None if cfg.mixed_precision == "no" else cfg.mixed_precision)
    device = accelerator.device

    model = build_model(
        model_name=cfg.model,
        patch_size=patch_size,
        in_channels=1,
        out_channels=len(CLASSES),
        feature_size=int(cfg.feature_size),
        use_checkpoint=bool(cfg.use_checkpoint),
    ).to(device)

    loss_fn = DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=1e-4)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    start_epoch = 1
    best_dice = -1.0
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_dice = float(ckpt.get("best_dice", -1.0))
    else:
        loaded = load_init_weights(model, cfg.init_from)
        if loaded:
            print(f"Loaded init weights tensors: {loaded}")

    if val_full_loader is None:
        model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)
    else:
        model, optimizer, train_loader, val_loader, val_full_loader = accelerator.prepare(
            model, optimizer, train_loader, val_loader, val_full_loader
        )

    if ckpt is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        move_optimizer_state_to_device(optimizer, accelerator.device)

    mlflow = None
    mlflow_active = False
    mlflow_run_id = ""
    if cfg.use_mlflow and accelerator.is_main_process:
        try:
            import mlflow as _mlflow
        except Exception:
            raise RuntimeError("MLflow not installed. Please: pip install mlflow")
        mlflow = _mlflow
        if cfg.mlflow_tracking_uri:
            mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
        if cfg.mlflow_experiment:
            mlflow.set_experiment(cfg.mlflow_experiment)

        resume_same_run = bool(cfg.mlflow_run_id)
        if resume_same_run:
            mlflow.start_run(run_id=cfg.mlflow_run_id)
        else:
            mlflow.start_run(run_name=cfg.mlflow_run_name)
        mlflow_run_id = mlflow.active_run().info.run_id

        git = git_info(os.getcwd())
        if git.get("sha"):
            mlflow.set_tag("git_sha", git["sha"])
        if git.get("branch"):
            mlflow.set_tag("git_branch", git["branch"])
        if "dirty" in git:
            mlflow.set_tag("git_dirty", str(bool(git["dirty"])))

        if not resume_same_run:
            for k, v in cfg.__dict__.items():
                if k in {"resume_from", "mlflow_run_id"}:
                    continue
                mlflow.log_param(k, v)

        if ckpt is not None:
            mlflow.set_tag("resume", "true")
            mlflow.set_tag("resume_from", str(cfg.resume_from))
        if cfg.init_from and ckpt is None:
            mlflow.set_tag("init_from", str(cfg.init_from))

        mlflow_active = True

    writer = None
    if cfg.use_tensorboard and accelerator.is_main_process:
        os.makedirs(cfg.tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=cfg.tb_dir)

    if cfg.epochs is None:
        raise ValueError("cfg.epochs cannot be None for a fresh run (only allowed in resume configs).")
    if start_epoch > cfg.epochs:
        if accelerator.is_main_process:
            print(f"[WARN] resume epoch ({start_epoch}) > cfg.epochs ({cfg.epochs}). Nothing to do.")
        if mlflow_active:
            mlflow.end_run()
        return

    best_path, last_path = checkpoint_paths(cfg.out)
    global_step = (start_epoch - 1) * max(1, len(train_loader))

    for epoch in range(start_epoch, int(cfg.epochs) + 1):
        t0 = time.time()

        avg_loss, global_step = train_one_epoch(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_loader=train_loader,
            accelerator=accelerator,
            device=device,
            epoch=epoch,
            log_steps=int(cfg.log_steps),
            writer=writer,
            global_step_base=global_step,
        )
        dice_patch = validate_dice(
            model=model,
            val_loader=val_loader,
            dice_metric=dice_metric,
            accelerator=accelerator,
            device=device,
            threshold=0.5,
        )
        dice_sw = None
        kaggle_summary = None
        if val_full_loader is not None and int(cfg.val_sw_every) > 0 and (epoch % int(cfg.val_sw_every) == 0):
            roi_size = (
                int(cfg.val_sw_patch_d or cfg.patch_d),
                int(cfg.val_sw_patch_h or cfg.patch_h),
                int(cfg.val_sw_patch_w or cfg.patch_w),
            )
            dice_sw = validate_sliding_window_dice(
                model=model,
                val_loader=val_full_loader,
                dice_metric=dice_metric,
                accelerator=accelerator,
                device=device,
                roi_size=roi_size,
                sw_batch_size=int(cfg.val_sw_batch_size),
                overlap=float(cfg.val_sw_overlap),
                threshold=0.5,
            )

            # Full-volume Kaggle-style score + save 16 sample images (like val_pipeline.ipynb).
            val_root = cfg.run_dir or os.path.dirname(cfg.out)
            full_out_dir = os.path.join(val_root, "val_full", f"epoch_{epoch:03d}")
            unwrapped = accelerator.unwrap_model(model)
            try:
                kaggle_summary = _run_full_val_kaggle_and_save(
                    model=unwrapped,
                    val_loader=val_full_loader,
                    accelerator=accelerator,
                    device=device,
                    epoch=epoch,
                    out_dir=full_out_dir,
                    roi_size=roi_size,
                    sw_batch_size=int(cfg.val_sw_batch_size),
                    overlap=float(cfg.val_sw_overlap),
                    threshold=float(cfg.val_full_threshold),
                    spatial_order=str(cfg.val_full_spatial_order),
                    normalize=str(cfg.val_full_normalize),
                    tta=bool(cfg.val_full_tta),
                    case_day_slices=case_day_slices,
                    rle_index=rle_index,
                    vis_samples=val_vis_samples,
                    vis_n=int(cfg.val_vis_n),
                    hausdorff_empty_score=float(cfg.val_full_hausdorff_empty_score),
                )
            except Exception as e:
                print(f"[WARN] Full-volume kaggle_score eval failed: {e}")
                kaggle_summary = None
        else:
            roi_size = (int(cfg.patch_d), int(cfg.patch_h), int(cfg.patch_w))

        dt = time.time() - t0
        if writer:
            writer.add_scalar("train/epoch_loss", float(avg_loss), epoch)
            writer.add_scalar("val/dice_patch", float(dice_patch), epoch)
            if dice_sw is not None:
                writer.add_scalar("val/dice_sw", float(dice_sw), epoch)
            if kaggle_summary and "final_score" in kaggle_summary:
                writer.add_scalar("val/kaggle_final_score", float(kaggle_summary["final_score"]), epoch)
                writer.add_scalar("val/kaggle_mean_dice", float(kaggle_summary.get("mean_dice_all", 0.0)), epoch)
                writer.add_scalar("val/kaggle_mean_hausdorff", float(kaggle_summary.get("mean_hausdorff_score", 0.0)), epoch)
                writer.add_text("val/kaggle_per_case_csv", str(kaggle_summary.get("out_per_case_csv", "")), epoch)
                writer.add_text("val/kaggle_csv_preview", str(kaggle_summary.get("tb_csv_preview_md", "")), epoch)
            writer.add_scalar("train/epoch_time_sec", float(dt), epoch)

            if (
                val_vis_samples
                and int(cfg.val_vis_every) > 0
                and (epoch % int(cfg.val_vis_every) == 0)
            ):
                unwrapped = accelerator.unwrap_model(model)
                _log_val_samples_to_tb(
                    writer=writer,
                    epoch=epoch,
                    samples=val_vis_samples,
                    case_day_slices=case_day_slices,
                    rle_index=rle_index,
                    model=unwrapped,
                    roi_size=roi_size,
                    sw_batch_size=int(cfg.val_sw_batch_size),
                    overlap=float(cfg.val_sw_overlap),
                    threshold=float(cfg.val_vis_threshold),
                )
        if mlflow_active:
            mlflow.log_metric("train/epoch_loss", float(avg_loss), step=epoch)
            mlflow.log_metric("val/dice_patch", float(dice_patch), step=epoch)
            if dice_sw is not None:
                mlflow.log_metric("val/dice_sw", float(dice_sw), step=epoch)
            if kaggle_summary and "final_score" in kaggle_summary:
                mlflow.log_metric("val/kaggle_final_score", float(kaggle_summary["final_score"]), step=epoch)
                mlflow.log_metric("val/kaggle_mean_dice", float(kaggle_summary.get("mean_dice_all", 0.0)), step=epoch)
                mlflow.log_metric("val/kaggle_mean_dice_nonempty", float(kaggle_summary.get("mean_dice_nonempty", 0.0)), step=epoch)
                mlflow.log_metric(
                    "val/kaggle_mean_hausdorff", float(kaggle_summary.get("mean_hausdorff_score", 0.0)), step=epoch
                )
                # Also persist the CSV/JSON for this epoch.
                try:
                    art_path = f"val_full/epoch_{epoch:03d}"
                    csv_path = str(kaggle_summary.get("out_per_case_csv", ""))
                    json_path = str(kaggle_summary.get("out_summary_json", ""))
                    if csv_path and os.path.exists(csv_path):
                        mlflow.log_artifact(csv_path, artifact_path=art_path)
                    if json_path and os.path.exists(json_path):
                        mlflow.log_artifact(json_path, artifact_path=art_path)
                except Exception as e:
                    print(f"[WARN] mlflow log_artifact failed: {e}")
            mlflow.log_metric("train/epoch_time_sec", float(dt), step=epoch)

        if accelerator.is_main_process:
            msg = f"Epoch {epoch:03d}/{cfg.epochs} | loss={avg_loss:.4f} | val_patch={dice_patch:.4f}"
            if dice_sw is not None:
                msg += f" | val_sw={dice_sw:.4f}"
            if kaggle_summary and "final_score" in kaggle_summary:
                msg += f" | val_kaggle={float(kaggle_summary['final_score']):.4f}"
            msg += f" | {dt:.1f}s"
            print(msg)

        best_metric = str(cfg.best_metric).lower()
        if best_metric == "sw":
            # Only update "best" on epochs where we actually computed SW metrics.
            primary = float(dice_sw) if dice_sw is not None else float("-inf")
        else:
            primary = float(dice_patch)

        if primary > best_dice and accelerator.is_main_process:
            best_dice = float(primary)
            unwrapped = accelerator.unwrap_model(model)
            torch.save(
                {
                    "model": unwrapped.state_dict(),
                    "optimizer": getattr(optimizer, "optimizer", optimizer).state_dict(),
                    "epoch": epoch,
                    "best_dice": best_dice,
                    "cfg": cfg.__dict__,
                    "mlflow_run_id": mlflow_run_id,
                },
                best_path,
            )

            if mlflow_active:
                mlflow.log_metric("best/val_dice", float(best_dice), step=epoch)
                mlflow.log_artifact(best_path, artifact_path="checkpoints")

        if accelerator.is_main_process and cfg.save_last_every > 0 and (epoch % int(cfg.save_last_every) == 0):
            unwrapped = accelerator.unwrap_model(model)
            torch.save(
                {
                    "model": unwrapped.state_dict(),
                    "optimizer": getattr(optimizer, "optimizer", optimizer).state_dict(),
                    "epoch": epoch,
                    "best_dice": best_dice,
                    "cfg": cfg.__dict__,
                    "mlflow_run_id": mlflow_run_id,
                },
                last_path,
            )

    if writer:
        writer.close()
    if accelerator.is_main_process:
        print(f"Done. Best val dice: {best_dice:.4f}")
    if mlflow_active:
        mlflow.end_run()
