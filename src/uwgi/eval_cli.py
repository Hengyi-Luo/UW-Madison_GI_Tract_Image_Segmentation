import argparse
import csv
import json
import os
import sys
import time
from typing import Optional

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from monai.inferers import sliding_window_inference

from .constants import CLASSES
from .data_utils import build_case_day_slices, build_rle_index, load_case_days
from .datasets import UWGI3DFullVolumeDataset
from .models import build_model


def _load_yaml(path: str):
    try:
        import yaml
    except Exception:
        print("PyYAML not installed. Please: pip install pyyaml", file=sys.stderr)
        raise
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _dice_sum_count_kaggle_2d(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Kaggle-style Dice for 2D images (slices), averaged over (image, class) pairs:
      - Dice is computed per slice per class over (H,W).
      - Dice is defined as 0 when both pred and target are empty (denom==0).

    pred/target: (B,C,D,H,W) float/bool.
    Returns (sum_dice, count_total, gt_non_empty) where:
      - count_total = B*C*D  (includes empty-empty pairs)
      - gt_non_empty counts pairs where target has any positive pixel.
    """
    pred = pred.float()
    target = target.float()
    inter = (pred * target).sum(dim=(-2, -1))  # (B,C,D)
    pred_sum = pred.sum(dim=(-2, -1))
    tgt_sum = target.sum(dim=(-2, -1))
    denom = pred_sum + tgt_sum
    dice = torch.where(denom > 0, (2.0 * inter + eps) / (denom + eps), torch.zeros_like(denom))
    count_total = torch.as_tensor(dice.numel(), device=dice.device)
    gt_non_empty = (tgt_sum > 0).sum()
    return dice.sum(), count_total, gt_non_empty


def _dice_sum_count_ignore_empty_3d(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Legacy/monai-style mean dice over 3D volumes, ignoring empty GT classes.

    pred/target: (B,C,D,H,W) float/bool.
    Returns (sum_dice, count_non_empty) across batch+channels where target has any positive voxel.
    """
    pred = pred.float()
    target = target.float()
    dims = tuple(range(2, pred.ndim))  # (D,H,W)
    inter = (pred * target).sum(dim=dims)  # (B,C)
    pred_sum = pred.sum(dim=dims)
    tgt_sum = target.sum(dim=dims)
    dice = (2.0 * inter + eps) / (pred_sum + tgt_sum + eps)
    not_empty = tgt_sum > 0
    dice = torch.where(not_empty, dice, torch.zeros_like(dice))
    return dice.sum(), not_empty.sum()


def _build_parser(defaults):
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--data_root", type=str, default=defaults.get("data_root"))
    p.add_argument("--weights", type=str, default=defaults.get("weights"))
    p.add_argument("--val_ids", type=str, default=defaults.get("val_ids"))
    p.add_argument("--ids_column", type=str, default=defaults.get("ids_column"))
    p.add_argument("--out", type=str, default=defaults.get("out"))  # json summary
    p.add_argument("--out_per_case", type=str, default=defaults.get("out_per_case"))  # csv per-case
    p.add_argument("--model", type=str, default=defaults.get("model"))  # optional override
    p.add_argument("--roi_d", type=int, default=defaults.get("roi_d"))
    p.add_argument("--roi_h", type=int, default=defaults.get("roi_h"))
    p.add_argument("--roi_w", type=int, default=defaults.get("roi_w"))
    p.add_argument("--sw_batch_size", type=int, default=defaults.get("sw_batch_size"))
    p.add_argument("--overlap", type=float, default=defaults.get("overlap"))
    p.add_argument("--threshold", type=float, default=defaults.get("threshold"))
    p.add_argument("--num_workers", type=int, default=defaults.get("num_workers"))
    p.add_argument("--mixed_precision", type=str, default=defaults.get("mixed_precision"))
    p.add_argument("--mode", type=str, default=defaults.get("mode"))  # sw | direct
    p.add_argument(
        "--metric",
        type=str,
        default=defaults.get("metric", "kaggle_dice_2d"),
        choices=["kaggle_dice_2d", "ignore_empty_3d"],
        help="kaggle_dice_2d: per-slice per-class dice (empty-empty=0). ignore_empty_3d: 3D dice ignoring empty GT classes.",
    )
    p.add_argument("--per_case", action="store_true", default=bool(defaults.get("per_case", False)))
    p.add_argument("--max_cases", type=int, default=defaults.get("max_cases"))
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

    data_root = args.data_root or "/home/jeremiah/github/UW-Madison_GI_Tract_Image_Segmentation/inputs"
    train_dir = os.path.join(data_root, "train")
    train_csv = os.path.join(data_root, "train.csv")
    weights = args.weights or "./outputs/train_run/best.pt"
    val_ids = args.val_ids

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"train.csv not found: {train_csv}")
    if not os.path.exists(weights):
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not val_ids:
        raise ValueError("--val_ids is required (file path with case_day or Kaggle id per line)")

    case_days = load_case_days(val_ids, column=args.ids_column)
    if args.max_cases is not None:
        case_days = case_days[: int(args.max_cases)]
    if len(case_days) == 0:
        raise ValueError(f"No valid case_day IDs loaded from: {val_ids}")

    if args.debug:
        case_days = case_days[: min(2, len(case_days))]

    print(f"Loaded val case_days: {len(case_days)}")

    print("Building slice index...")
    case_day_slices = build_case_day_slices(train_dir)
    missing = [cd for cd in case_days if cd not in case_day_slices]
    if missing:
        print(f"[WARN] {len(missing)} case_days not found under train/: e.g. {missing[:5]}", file=sys.stderr)
    case_days = [cd for cd in case_days if cd in case_day_slices]
    if len(case_days) == 0:
        raise ValueError("All provided case_days are missing from train/ directory; cannot evaluate.")

    print("Building RLE index...")
    rle_index = build_rle_index(train_csv)

    ds = UWGI3DFullVolumeDataset(
        case_days=case_days,
        case_day_slices=case_day_slices,
        rle_index=rle_index,
        cache_in_ram=False,
        verbose_shape_fix=False,
    )
    num_workers = int(args.num_workers or 2)
    if args.debug:
        num_workers = 0
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

    ckpt = torch.load(weights, map_location="cpu")
    ckpt_cfg = ckpt.get("cfg", {}) if isinstance(ckpt, dict) else {}
    model_name = (args.model or ckpt_cfg.get("model", "unet") or "unet").lower()
    if "model" in ckpt_cfg and args.model and str(ckpt_cfg.get("model")).lower() != model_name:
        print(f"[WARN] overriding checkpoint model={ckpt_cfg.get('model')} with --model={model_name}", file=sys.stderr)

    patch_size = (
        int(ckpt_cfg.get("patch_d", 96)),
        int(ckpt_cfg.get("patch_h", 224)),
        int(ckpt_cfg.get("patch_w", 224)),
    )

    model = build_model(
        model_name=model_name,
        patch_size=patch_size,
        in_channels=1,
        out_channels=len(CLASSES),
        feature_size=int(ckpt_cfg.get("feature_size", 48)),
        use_checkpoint=bool(ckpt_cfg.get("use_checkpoint", False)),
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    mixed_precision = args.mixed_precision or ckpt_cfg.get("mixed_precision", "no") or "no"
    accelerator = Accelerator(mixed_precision=None if mixed_precision == "no" else mixed_precision)
    device = accelerator.device

    model, dl = accelerator.prepare(model.to(device), dl)

    roi_size = (
        int(args.roi_d or patch_size[0]),
        int(args.roi_h or patch_size[1]),
        int(args.roi_w or patch_size[2]),
    )
    sw_batch_size = int(args.sw_batch_size or 1)
    overlap = float(args.overlap if args.overlap is not None else 0.25)
    thr = float(args.threshold if args.threshold is not None else 0.5)
    mode = (args.mode or "sw").lower()

    metric = str(args.metric or "kaggle_dice_2d")
    dice_sum = 0.0
    count_total = 0
    gt_non_empty_total = 0
    per_case_rows: list[dict] = []

    with torch.no_grad():
        for x, y, case_day in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with accelerator.autocast():
                if mode == "direct":
                    logits = model(x)
                elif mode == "sw":
                    logits = sliding_window_inference(x, roi_size, sw_batch_size, model, overlap=overlap)
                else:
                    raise ValueError(f"Unknown --mode={mode}. Supported: sw, direct")

                prob = torch.sigmoid(logits)
                pred = (prob > thr).float()

            if metric == "kaggle_dice_2d":
                s, c_total, gt_ne = _dice_sum_count_kaggle_2d(pred, y)
                dice_sum += float(s.item())
                count_total += int(c_total.item())
                gt_non_empty_total += int(gt_ne.item())
                case_dice = float(s.item()) / max(1, int(c_total.item()))
            elif metric == "ignore_empty_3d":
                s, c_ne = _dice_sum_count_ignore_empty_3d(pred, y)
                dice_sum += float(s.item())
                gt_non_empty_total += int(c_ne.item())
                case_dice = float(s.item()) / max(1, int(c_ne.item()))
            else:
                raise ValueError(f"Unknown metric={metric}")

            if accelerator.is_main_process and (args.out_per_case or args.out):
                name = case_day[0] if isinstance(case_day, (list, tuple)) else str(case_day)
                row = {"case_day": name, "dice": case_dice}
                if metric == "kaggle_dice_2d":
                    row.update(
                        {
                            "count_total": int(c_total.item()),
                            "gt_non_empty": int(gt_ne.item()),
                            "num_slices": int(y.shape[2]),
                            "num_classes": int(y.shape[1]),
                        }
                    )
                else:
                    row.update(
                        {
                            "gt_non_empty": int(c_ne.item()),
                            "num_classes": int(y.shape[1]),
                        }
                    )
                per_case_rows.append(row)
            if args.per_case and accelerator.is_main_process:
                name = case_day[0] if isinstance(case_day, (list, tuple)) else str(case_day)
                print(f"{name}\t{case_dice:.6f}")

    denom = count_total if metric == "kaggle_dice_2d" else gt_non_empty_total
    mean_dice = dice_sum / max(1, denom)
    if accelerator.is_main_process:
        print(
            "Done."
            f" mean_dice={mean_dice:.6f}"
            f" count_total={count_total}"
            f" gt_non_empty={gt_non_empty_total}"
            f" mode={mode} roi={roi_size} thr={thr}"
            f" metric={metric}"
        )
        if args.out_per_case:
            out_csv = str(args.out_per_case)
            os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
            with open(out_csv, "w", encoding="utf-8", newline="") as f:
                fieldnames = ["case_day", "dice"]
                if metric == "kaggle_dice_2d":
                    fieldnames += ["count_total", "gt_non_empty", "num_slices", "num_classes"]
                else:
                    fieldnames += ["gt_non_empty", "num_classes"]
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(per_case_rows)
            print(f"Saved per-case: {out_csv}")

        if args.out:
            out_json = str(args.out)
            os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
            payload = {
                "timestamp": int(time.time()),
                "data_root": data_root,
                "weights": weights,
                "model": model_name,
                "metric": metric,
                "mode": mode,
                "roi_size": list(roi_size),
                "sw_batch_size": int(sw_batch_size),
                "overlap": float(overlap),
                "threshold": float(thr),
                "mean_dice": float(mean_dice),
                "count_total": int(count_total),
                "gt_non_empty": int(gt_non_empty_total),
                "num_case_days": int(len(per_case_rows) or len(case_days)),
            }
            if per_case_rows:
                payload["per_case"] = per_case_rows
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            print(f"Saved summary: {out_json}")


if __name__ == "__main__":
    main()
