import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from monai.inferers import sliding_window_inference

from .constants import CLASS2IDX
from .data_utils import build_case_day_slices
from .datasets import TestVolumeDataset
from .models import build_model
from .rle import rle_encode


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
    p.add_argument("--weights", type=str, default=defaults.get("weights"))
    p.add_argument("--out_csv", type=str, default=defaults.get("out_csv"))
    p.add_argument("--test_dir", type=str, default=defaults.get("test_dir"))
    p.add_argument("--sample_submission", type=str, default=defaults.get("sample_submission"))
    p.add_argument("--sw_patch_d", type=int, default=defaults.get("sw_patch_d"))
    p.add_argument("--sw_patch_h", type=int, default=defaults.get("sw_patch_h"))
    p.add_argument("--sw_patch_w", type=int, default=defaults.get("sw_patch_w"))
    p.add_argument("--sw_batch_size", type=int, default=defaults.get("sw_batch_size"))
    p.add_argument("--threshold", type=float, default=defaults.get("threshold"))
    p.add_argument("--num_workers", type=int, default=defaults.get("num_workers"))
    p.add_argument("--max_cases", type=int, default=defaults.get("max_cases"))
    p.add_argument("--max_slices", type=int, default=defaults.get("max_slices"))
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
    test_dir = args.test_dir or os.path.join(data_root, "test")
    weights = args.weights or "./outputs/train_run/best.pt"
    out_csv = args.out_csv or "./outputs/submission.csv"
    sample_submission = args.sample_submission

    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    if not os.path.exists(weights):
        raise FileNotFoundError(f"Weights not found: {weights}")

    max_cases = args.max_cases
    max_slices = args.max_slices
    num_workers = args.num_workers

    if args.debug:
        if max_cases is None:
            max_cases = 1
        if max_slices is None:
            max_slices = 8
        num_workers = 0

    case_day_slices = build_case_day_slices(test_dir)
    ds = TestVolumeDataset(case_day_slices, max_cases=max_cases, max_slices=max_slices)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers or 2)

    ckpt = torch.load(weights, map_location="cpu")
    cfg = ckpt.get("cfg", {}) if isinstance(ckpt, dict) else {}
    model_name = cfg.get("model", "unet")
    patch_size = (
        int(cfg.get("patch_d", args.sw_patch_d or 96)),
        int(cfg.get("patch_h", args.sw_patch_h or 224)),
        int(cfg.get("patch_w", args.sw_patch_w or 224)),
    )
    model = build_model(
        model_name=model_name,
        patch_size=patch_size,
        in_channels=1,
        out_channels=3,
        feature_size=int(cfg.get("feature_size", 48)),
        use_checkpoint=bool(cfg.get("use_checkpoint", False)),
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    roi_size = (
        args.sw_patch_d or patch_size[0],
        args.sw_patch_h or patch_size[1],
        args.sw_patch_w or patch_size[2],
    )
    sw_batch_size = args.sw_batch_size or 1
    thr = args.threshold if args.threshold is not None else 0.5

    outputs = []
    with torch.no_grad():
        for x, case_day, slice_indices in dl:
            x = x.to(device)  # (1,1,D,H,W)
            pred = sliding_window_inference(x, roi_size, sw_batch_size, model)
            pred = torch.sigmoid(pred)[0].cpu().numpy()  # (3,D,H,W)
            pred = (pred > thr).astype(np.uint8)

            for z, slice_idx in enumerate(slice_indices[0]):
                id_name = f"{case_day[0]}_slice_{slice_idx:04d}"
                for cls_name, cls_i in CLASS2IDX.items():
                    mask = pred[cls_i, z]
                    outputs.append([id_name, cls_name, rle_encode(mask)])

    pred_df = pd.DataFrame(outputs, columns=["id", "class", "predicted"])
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    if sample_submission:
        sub_df = pd.read_csv(sample_submission)
        if "predicted" in sub_df.columns:
            sub_df = sub_df.drop(columns=["predicted"])
        sub_df = sub_df.merge(pred_df, on=["id", "class"], how="left")
        sub_df["predicted"] = sub_df["predicted"].fillna("")
        sub_df.to_csv(out_csv, index=False)
    else:
        pred_df.to_csv(out_csv, index=False)

    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
