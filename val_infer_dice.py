#!/usr/bin/env python3
"""
Run full-volume inference on val set, compute Dice per volume, and save sample visualizations.

Example:
  python scripts/val_infer_dice.py \
    --weights run01/best.pt \
    --val_ids inputs/splits/val_case_days.csv \
    --out_per_case outputs/val_eval/eval_per_case.csv \
    --vis_samples_csv inputs/splits/val_vis_samples.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Iterable

import cv2
import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.networks.layers import Norm
from monai.networks.nets import Unet
from monai.transforms import Compose, EnsureChannelFirstd, EnsureTyped, ScaleIntensityd, SpatialPadd

from src.constants import CLASSES  # noqa: E402
from src.data_utils import build_case_day_slices, build_rle_index, load_case_days  # noqa: E402
from src.datasets import LoadCaseDayVolumed  # noqa: E402


def _dice_per_class(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    pred/target: (C,D,H,W) float/bool.
    Returns per-class Dice (C,).
    """
    pred = pred.float()
    target = target.float()
    dims = tuple(range(1, pred.ndim))
    inter = (pred * target).sum(dim=dims)
    denom = pred.sum(dim=dims) + target.sum(dim=dims)
    return (2.0 * inter + eps) / (denom + eps)


def _build_model(device: torch.device) -> torch.nn.Module:
    model = Unet(
        spatial_dims=3,
        in_channels=1,
        out_channels=len(CLASSES),
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=0.2,
        norm=Norm.BATCH,
    ).to(device)
    return model


def _get_val_transforms(
    *,
    patch_d: int,
    patch_h: int,
    patch_w: int,
    case_day_slices,
    rle_index,
):
    return Compose(
        [
            LoadCaseDayVolumed(
                keys=["case_day"],
                case_day_slices=case_day_slices,
                rle_index=rle_index,
            ),
            EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            ScaleIntensityd(
                keys=["image"],
                minv=0.0,
                maxv=1.0,
            ),
            SpatialPadd(
                keys=["image", "label"],
                spatial_size=(int(patch_d), int(patch_h), int(patch_w)),
                mode="constant",
            ),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def _load_vis_samples_csv(path: str) -> list[dict]:
    if not path:
        return []
    if not os.path.exists(path):
        raise FileNotFoundError(f"vis_samples_csv not found: {path}")
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            case_day = str(row.get("case_day") or row.get("case_day_id") or row.get("id") or "").strip()
            if not case_day:
                continue
            try:
                slice_pos = int(row.get("slice_pos", 0))
            except Exception:
                slice_pos = 0
            try:
                slice_idx = int(row.get("slice_idx", -1))
            except Exception:
                slice_idx = -1
            rows.append(
                {
                    "case_day": case_day,
                    "slice_pos": int(slice_pos),
                    "slice_idx": int(slice_idx),
                }
            )
    return rows


def _plot_case_day_slice_gt_pred(
    case_day: str,
    slice_pos: int,
    case_day_slices: dict[str, list[str]],
    pred_vol: np.ndarray,
    gt_vol: np.ndarray,
    *,
    classes: Iterable[str],
    figsize: tuple[int, int] = (14, 7),
    overlay_figsize: tuple[int, int] = (12, 6),
    alpha: float = 0.4,
):
    import matplotlib.pyplot as plt

    cls_list = list(classes)
    slice_paths = case_day_slices[case_day]
    if slice_pos < 0 or slice_pos >= len(slice_paths):
        raise IndexError(f"slice_pos={slice_pos} out of range for {case_day}")

    slice_path = slice_paths[slice_pos]
    img = cv2.imread(slice_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {slice_path}")

    pred2d = {cls_list[i]: pred_vol[i, slice_pos] for i in range(len(cls_list))}
    gt2d = {cls_list[i]: gt_vol[i, slice_pos] for i in range(len(cls_list))}

    cmaps = ["Reds", "Greens", "Blues"]
    fig_grid = plt.figure(figsize=figsize)
    ax = plt.subplot(2, 4, 1)
    ax.set_title("image")
    ax.imshow(img, cmap="gray")
    ax.axis("off")

    for i, cls in enumerate(cls_list):
        ax = plt.subplot(2, 4, 2 + i)
        ax.set_title(f"GT: {cls}")
        ax.imshow(gt2d[cls], cmap=cmaps[i])
        ax.axis("off")

    ax = plt.subplot(2, 4, 5)
    ax.set_title("image")
    ax.imshow(img, cmap="gray")
    ax.axis("off")

    for i, cls in enumerate(cls_list):
        ax = plt.subplot(2, 4, 6 + i)
        ax.set_title(f"Pred: {cls}")
        ax.imshow(pred2d[cls], cmap=cmaps[i])
        ax.axis("off")

    plt.tight_layout()

    fig_overlay = plt.figure(figsize=overlay_figsize)
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_title("overlay: GT")
    ax1.imshow(img, cmap="gray")
    for i, cls in enumerate(cls_list):
        ax1.imshow(gt2d[cls], alpha=float(alpha), cmap=cmaps[i])
    ax1.axis("off")

    ax2 = plt.subplot(1, 2, 2)
    ax2.set_title("overlay: Pred")
    ax2.imshow(img, cmap="gray")
    for i, cls in enumerate(cls_list):
        ax2.imshow(pred2d[cls], alpha=float(alpha), cmap=cmaps[i])
    ax2.axis("off")

    plt.tight_layout()
    return fig_grid, fig_overlay


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./inputs")
    p.add_argument("--weights", type=str, default="outputs/20260206-060400_Unet3D_c0/best.pt")
    p.add_argument("--val_ids", type=str, default="inputs/splits/c0_val_case_days.csv")
    p.add_argument("--out_per_case", type=str, default="outputs/20260206-060400_Unet3D_c0/val_eval/eval_per_case.csv")
    p.add_argument("--out_summary", type=str, default="outputs/20260206-060400_Unet3D_c0/val_eval/eval_summary.json")
    p.add_argument("--roi_d", type=int, default=96)
    p.add_argument("--roi_h", type=int, default=224)
    p.add_argument("--roi_w", type=int, default=224)
    p.add_argument("--sw_batch_size", type=int, default=4)
    p.add_argument("--overlap", type=float, default=0.25)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_cases", type=int, default=None)
    p.add_argument("--vis_samples_csv", type=str, default="./inputs/splits/val_vis_samples.csv")
    p.add_argument("--vis_out_dir", type=str, default="")
    p.add_argument("--vis_n", type=int, default=None)
    p.add_argument("--device", type=str, default="")
    p.add_argument(
        "--unsafe_load",
        action="store_true",
        help="Set weights_only=False when loading checkpoint (unsafe unless you trust the source).",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    data_root = args.data_root
    train_dir = os.path.join(data_root, "train")
    train_csv = os.path.join(data_root, "train.csv")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"train.csv not found: {train_csv}")
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    if not os.path.exists(args.val_ids):
        raise FileNotFoundError(f"val_ids not found: {args.val_ids}")

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Building slice index...")
    case_day_slices = build_case_day_slices(train_dir)
    print("Building RLE index...")
    rle_index = build_rle_index(train_csv)

    case_days = load_case_days(args.val_ids)
    if args.max_cases is not None:
        case_days = case_days[: int(args.max_cases)]
    case_days = [cd for cd in case_days if cd in case_day_slices]
    if not case_days:
        raise ValueError("No valid case_days for evaluation after filtering.")
    print(f"Val case_days: {len(case_days)}")

    val_files = [{"case_day": cd} for cd in case_days]
    val_transforms = _get_val_transforms(
        patch_d=int(args.roi_d),
        patch_h=int(args.roi_h),
        patch_w=int(args.roi_w),
        case_day_slices=case_day_slices,
        rle_index=rle_index,
    )
    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )

    model = _build_model(device)
    try:
        ckpt = torch.load(args.weights, map_location=device, weights_only=not bool(args.unsafe_load))
    except Exception as e:
        # PyTorch 2.6+ safe deserialization: allowlist numpy reconstruct if needed.
        if "Weights only load failed" in str(e):
            import numpy as _np

            try:
                from torch.serialization import safe_globals
            except Exception:
                safe_globals = None

            if safe_globals is not None:
                with safe_globals([_np._core.multiarray._reconstruct]):
                    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
            else:
                raise
        else:
            raise
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        raise ValueError("Unrecognized checkpoint format: expected model_state_dict or model keys.")
    model.eval()

    roi_size = (int(args.roi_d), int(args.roi_h), int(args.roi_w))
    sw_batch_size = int(args.sw_batch_size)
    overlap = float(args.overlap)
    thr = float(args.threshold)

    samples = _load_vis_samples_csv(args.vis_samples_csv) if args.vis_samples_csv else []
    if args.vis_n is not None:
        samples = samples[: int(args.vis_n)]
    sample_case_days = {str(s.get("case_day")) for s in samples}
    cache_vols: dict[str, dict] = {}

    per_case_rows: list[dict] = []
    dice_sum = torch.zeros(len(CLASSES), dtype=torch.float64)

    with torch.no_grad():
        for batch in val_loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device).float()
            case_day = batch["case_day"][0]

            logits = sliding_window_inference(
                x,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=overlap,
            )
            probs = torch.sigmoid(logits)
            pred = (probs > thr).float()

            pred_c = pred[0]
            gt_c = y[0]
            dice_c = _dice_per_class(pred_c, gt_c)
            dice_mean = float(dice_c.mean().item())
            dice_sum += dice_c.double().cpu()

            per_case_rows.append(
                {
                    "case_day": str(case_day),
                    "dice_mean": dice_mean,
                    "dice_large_bowel": float(dice_c[0].item()),
                    "dice_small_bowel": float(dice_c[1].item()),
                    "dice_stomach": float(dice_c[2].item()),
                    "num_slices": int(gt_c.shape[1]),
                }
            )

            if case_day in sample_case_days and case_day not in cache_vols:
                cache_vols[case_day] = {
                    "pred": pred_c.detach().cpu().to(torch.uint8).numpy(),
                    "gt": gt_c.detach().cpu().to(torch.uint8).numpy(),
                }

    mean_dice_per_class = (dice_sum / max(1, len(per_case_rows))).tolist()
    mean_dice = float(sum(mean_dice_per_class) / len(mean_dice_per_class))

    os.makedirs(os.path.dirname(args.out_per_case) or ".", exist_ok=True)
    with open(args.out_per_case, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "case_day",
                "dice_mean",
                "dice_large_bowel",
                "dice_small_bowel",
                "dice_stomach",
                "num_slices",
            ],
        )
        w.writeheader()
        w.writerows(per_case_rows)
    print(f"Saved per-case: {args.out_per_case}")

    os.makedirs(os.path.dirname(args.out_summary) or ".", exist_ok=True)
    summary = {
        "data_root": data_root,
        "weights": args.weights,
        "roi_size": list(roi_size),
        "sw_batch_size": int(sw_batch_size),
        "overlap": float(overlap),
        "threshold": float(thr),
        "num_case_days": int(len(per_case_rows)),
        "mean_dice": float(mean_dice),
        "mean_dice_per_class": [float(x) for x in mean_dice_per_class],
    }
    with open(args.out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved summary: {args.out_summary}")

    if samples:
        vis_out_dir = args.vis_out_dir
        if not vis_out_dir:
            base_dir = os.path.dirname(args.out_per_case) or "."
            vis_out_dir = os.path.join(base_dir, "samples")
        os.makedirs(vis_out_dir, exist_ok=True)
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"[WARN] Skipping sample saving (matplotlib not available): {e}")
            return

        for i, s in enumerate(samples):
            case_day = str(s.get("case_day"))
            if case_day not in case_day_slices:
                continue
            slice_pos = int(s.get("slice_pos", 0))
            slice_idx = int(s.get("slice_idx", -1))
            vols = cache_vols.get(case_day)
            if vols is None:
                continue
            fig_grid, fig_overlay = _plot_case_day_slice_gt_pred(
                case_day,
                slice_pos,
                case_day_slices,
                pred_vol=vols["pred"],
                gt_vol=vols["gt"],
                classes=CLASSES,
            )
            base = f"{i:02d}_{case_day}_slicepos_{slice_pos:03d}_sliceidx_{slice_idx:04d}"
            fig_grid.savefig(os.path.join(vis_out_dir, f"{base}__grid.png"), dpi=150, bbox_inches="tight")
            fig_overlay.savefig(os.path.join(vis_out_dir, f"{base}__overlay.png"), dpi=150, bbox_inches="tight")
            plt.close(fig_grid)
            plt.close(fig_overlay)
        print(f"Saved samples: {vis_out_dir}")


if __name__ == "__main__":
    main()
