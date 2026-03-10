#!/usr/bin/env python3
"""
Inference script: load a trained 3D U-Net checkpoint, run full-volume inference on a split of case_days,
and write a Kaggle-style RLE CSV (same schema as inputs/train.csv) named submit.csv under the model's
outputs directory.

Also optionally computes case_day-level Dice evaluation and summary (when `evaluate: true`) by decoding
GT masks from inputs/train.csv.

Usage:
  python scripts/infer_unet3d.py --config configs/infer_unet3d.yaml

Default output:
  <weights_parent>/infer/submit.csv
  <weights_parent>/infer/eval_per_case.csv
  <weights_parent>/infer/eval_summary.json
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import csv
import json
import math
import os
import sys
from typing import Any

import torch
import yaml
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.networks.layers import Norm
from monai.networks.nets import Unet
from monai.transforms import Compose, EnsureChannelFirstd, EnsureTyped, ScaleIntensityd, SpatialPadd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)


from src.constants import CLASSES  # noqa: E402
from src.data_utils import build_case_day_slices, build_rle_index, load_case_days  # noqa: E402
from src.datasets import LoadCaseDayImaged, LoadCaseDayVolumed  # noqa: E402
from src.metrics import SegmentationMetricAccumulator  # noqa: E402
from src.rle import rle_encode  # noqa: E402


@dataclass(frozen=True)
class InferCfg:
    data_root: str = "inputs"
    ids_csv: str = "inputs/splits/val_case_days.csv"
    train_csv: str = "inputs/train.csv"

    weights: str = ""
    output_dir: str = ""  # default: <weights_parent>/infer

    out_csv_name: str = "submit.csv"

    evaluate: bool = True
    eval_per_case_name: str = "eval_per_case.csv"
    eval_summary_name: str = "eval_summary.json"

    roi_d: int = 96
    roi_h: int = 224
    roi_w: int = 224
    sw_batch_size: int = 4
    overlap: float = 0.25
    threshold: float = 0.5

    num_workers: int = 0
    device: str = ""

    unsafe_load: bool = False

    @property
    def roi_size(self) -> tuple[int, int, int]:
        return (int(self.roi_d), int(self.roi_h), int(self.roi_w))

    @property
    def eval_enabled(self) -> bool:
        return bool(self.evaluate)


def _load_cfg(path: str) -> InferCfg:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError("Config YAML must be a mapping at the top level.")

    known = set(InferCfg().__dict__.keys())
    unknown = sorted([k for k in raw.keys() if k not in known])
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")

    merged = {**InferCfg().__dict__, **raw}
    return InferCfg(**merged)


def _build_model(device: torch.device) -> torch.nn.Module:
    return Unet(
        spatial_dims=3,
        in_channels=1,
        out_channels=len(CLASSES),
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=0.2,
        norm=Norm.BATCH,
    ).to(device)


def _get_infer_transforms(*, patch_d: int, patch_h: int, patch_w: int, case_day_slices):
    return Compose(
        [
            LoadCaseDayImaged(keys=["case_day"], case_day_slices=case_day_slices),
            EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            SpatialPadd(keys=["image"], spatial_size=(int(patch_d), int(patch_h), int(patch_w)), mode="constant"),
            EnsureTyped(keys=["image"]),
        ]
    )


def _get_eval_transforms(*, patch_d: int, patch_h: int, patch_w: int, case_day_slices, rle_index):
    return Compose(
        [
            LoadCaseDayVolumed(keys=["case_day"], case_day_slices=case_day_slices, rle_index=rle_index),
            EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            SpatialPadd(
                keys=["image", "label"],
                spatial_size=(int(patch_d), int(patch_h), int(patch_w)),
                mode="constant",
            ),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def _load_checkpoint(weights_path: str, device: torch.device, *, unsafe_load: bool) -> dict[str, Any]:
    try:
        ckpt = torch.load(weights_path, map_location=device, weights_only=not bool(unsafe_load))
    except Exception as e:
        if "Weights only load failed" in str(e):
            import numpy as _np

            try:
                from torch.serialization import safe_globals
            except Exception:
                safe_globals = None

            if safe_globals is not None:
                with safe_globals([_np._core.multiarray._reconstruct]):
                    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
            else:
                raise
        else:
            raise
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint must be a dict.")
    return ckpt


def _resolve_output_dir(cfg: InferCfg) -> Path:
    if cfg.output_dir:
        return Path(cfg.output_dir)
    if not cfg.weights:
        raise ValueError("weights must be set (either in config or via --weights).")
    return Path(cfg.weights).resolve().parent / "infer"


def _write_submit_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "class", "segmentation"])
        w.writeheader()
        w.writerows(rows)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/infer_unet3d.yaml")
    p.add_argument("--weights", type=str, default="", help="Override cfg.weights")
    p.add_argument("--ids_csv", type=str, default="", help="Override cfg.ids_csv")
    p.add_argument("--output_dir", type=str, default="", help="Override cfg.output_dir")
    p.add_argument("--device", type=str, default="", help="Override cfg.device")
    p.add_argument("--num_workers", type=int, default=None, help="Override cfg.num_workers")
    p.add_argument(
        "--unsafe_load",
        action="store_true",
        help="Set weights_only=False when loading checkpoint (unsafe unless you trust the source).",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    cfg = _load_cfg(args.config) if args.config and os.path.exists(args.config) else InferCfg()
    if args.weights:
        cfg = InferCfg(**{**asdict(cfg), "weights": args.weights})
    if args.ids_csv:
        cfg = InferCfg(**{**asdict(cfg), "ids_csv": args.ids_csv})
    if args.output_dir:
        cfg = InferCfg(**{**asdict(cfg), "output_dir": args.output_dir})
    if args.device:
        cfg = InferCfg(**{**asdict(cfg), "device": args.device})
    if args.num_workers is not None:
        cfg = InferCfg(**{**asdict(cfg), "num_workers": int(args.num_workers)})
    if args.unsafe_load:
        cfg = InferCfg(**{**asdict(cfg), "unsafe_load": True})

    if not cfg.weights:
        raise ValueError("Missing weights. Set it in config or pass --weights.")
    if not os.path.exists(cfg.weights):
        raise FileNotFoundError(f"Weights not found: {cfg.weights}")
    if not os.path.exists(cfg.ids_csv):
        raise FileNotFoundError(f"ids_csv not found: {cfg.ids_csv}")

    out_dir = _resolve_output_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "config.resolved.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n", encoding="utf-8")
    if args.config and os.path.exists(args.config):
        (out_dir / "config.yaml").write_text(Path(args.config).read_text(encoding="utf-8"), encoding="utf-8")

    device = torch.device(cfg.device) if cfg.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_root = cfg.data_root
    train_dir = os.path.join(data_root, "train")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    print("Building slice index...")
    case_day_slices = build_case_day_slices(train_dir)
    case_days = load_case_days(cfg.ids_csv)
    case_days = [cd for cd in case_days if cd in case_day_slices]
    if not case_days:
        raise ValueError("No valid case_days for inference after filtering.")
    print(f"Case_days: {len(case_days)}")

    infer_files = [{"case_day": cd} for cd in case_days]
    metric_acc = SegmentationMetricAccumulator(class_names=CLASSES, threshold=float(cfg.threshold))

    if cfg.eval_enabled:
        if not os.path.exists(cfg.train_csv):
            raise FileNotFoundError(f"train_csv not found (needed for evaluation): {cfg.train_csv}")
        print("Building RLE index (for evaluation)...")
        rle_index = build_rle_index(cfg.train_csv)
        infer_transforms = _get_eval_transforms(
            patch_d=int(cfg.roi_d),
            patch_h=int(cfg.roi_h),
            patch_w=int(cfg.roi_w),
            case_day_slices=case_day_slices,
            rle_index=rle_index,
        )
    else:
        infer_transforms = _get_infer_transforms(
            patch_d=int(cfg.roi_d),
            patch_h=int(cfg.roi_h),
            patch_w=int(cfg.roi_w),
            case_day_slices=case_day_slices,
        )

    infer_ds = Dataset(data=infer_files, transform=infer_transforms)

    def _make_loader(num_workers: int) -> DataLoader:
        return DataLoader(
            infer_ds,
            batch_size=1,
            shuffle=False,
            num_workers=int(num_workers),
            pin_memory=(device.type == "cuda"),
        )

    infer_loader = _make_loader(int(cfg.num_workers))
    try:
        _ = iter(infer_loader)
    except PermissionError:
        if int(cfg.num_workers) != 0:
            print("[WARN] DataLoader multiprocessing not permitted; retrying with num_workers=0")
            infer_loader = _make_loader(0)

    model = _build_model(device)
    ckpt = _load_checkpoint(cfg.weights, device, unsafe_load=bool(cfg.unsafe_load))
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        raise ValueError("Unrecognized checkpoint format: expected model_state_dict or model keys.")
    model.eval()

    roi_size = cfg.roi_size
    sw_batch_size = int(cfg.sw_batch_size)
    overlap = float(cfg.overlap)
    thr = float(cfg.threshold)

    rows: list[dict[str, str]] = []
    with torch.no_grad():
        for batch in infer_loader:
            x = batch["image"].to(device)
            case_day = str(batch["case_day"][0])
            orig_d, orig_h, orig_w = [int(t.item()) for t in batch["orig_shape"]]
            slice_idxs = [int(t.item()) for t in batch["slice_idxs"]]

            logits = sliding_window_inference(
                x,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=overlap,
            )
            probs = torch.sigmoid(logits)
            pred = (probs > thr).to(torch.uint8)  # (1,C,D,H,W)

            pred_cdhw = pred[0, :, :orig_d, :orig_h, :orig_w].detach().cpu().numpy()

            if cfg.eval_enabled:
                y = batch["label"].to(device).to(torch.uint8)  # (1,C,D,H,W)
                metric_acc.update(
                    pred[:, :, :orig_d, :orig_h, :orig_w].float(),
                    y[:, :, :orig_d, :orig_h, :orig_w].float(),
                    case_ids=[case_day],
                    num_slices=[int(orig_d)],
                )

            for z in range(int(orig_d)):
                slice_idx = int(slice_idxs[z]) if z < len(slice_idxs) else (z + 1)
                _id = f"{case_day}_slice_{slice_idx:04d}"
                for ci, cls in enumerate(CLASSES):
                    m2d = pred_cdhw[ci, z]
                    seg = rle_encode(m2d)
                    rows.append({"id": _id, "class": str(cls), "segmentation": str(seg)})

    out_csv = out_dir / str(cfg.out_csv_name)
    _write_submit_csv(out_csv, rows)

    if cfg.eval_enabled:
        per_case_rows = metric_acc.per_case_rows()
        eval_summary_metrics = metric_acc.summary()
        eval_per_case_path = out_dir / str(cfg.eval_per_case_name)
        with open(eval_per_case_path, "w", encoding="utf-8", newline="") as f:
            fieldnames = list(per_case_rows[0].keys()) if per_case_rows else ["case_day", "num_slices", "dice_mean"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(per_case_rows)

        eval_summary = {
            "data_root": str(Path(cfg.data_root).resolve()),
            "train_csv": str(Path(cfg.train_csv).resolve()),
            "weights": str(Path(cfg.weights).resolve()),
            "ids_csv": str(Path(cfg.ids_csv).resolve()),
            "protocol": "docs/METRICS_PROTOCOL.md",
            "hd95_units": "voxels",
            "num_case_days": int(len(per_case_rows)),
            **eval_summary_metrics,
            "mean_dice": float(eval_summary_metrics["dice_mean"]),
            "mean_dice_per_class": [float(eval_summary_metrics[f"dice_{cls_name}"]) for cls_name in CLASSES],
            "threshold": float(thr),
            "roi_size": list(roi_size),
            "sw_batch_size": int(sw_batch_size),
            "overlap": float(overlap),
        }
        (out_dir / str(cfg.eval_summary_name)).write_text(
            json.dumps(_json_safe(eval_summary), indent=2) + "\n",
            encoding="utf-8",
        )

    summary = {
        "weights": str(Path(cfg.weights).resolve()),
        "ids_csv": str(Path(cfg.ids_csv).resolve()),
        "num_case_days": int(len(case_days)),
        "num_rows": int(len(rows)),
        "threshold": float(thr),
        "roi_size": list(roi_size),
        "sw_batch_size": int(sw_batch_size),
        "overlap": float(overlap),
        "output_csv": str(out_csv),
    }
    (out_dir / "infer_summary.json").write_text(json.dumps(_json_safe(summary), indent=2) + "\n", encoding="utf-8")

    print(f"Wrote submission CSV: {out_csv}")
    if cfg.eval_enabled:
        print(f"Wrote eval per-case : {out_dir / str(cfg.eval_per_case_name)}")
        print(f"Wrote eval summary  : {out_dir / str(cfg.eval_summary_name)}")
    print(f"Wrote summary        : {out_dir / 'infer_summary.json'}")


if __name__ == "__main__":
    main()
