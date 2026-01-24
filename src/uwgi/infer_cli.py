import argparse
import os
import sys
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.transforms import Compose, EnsureChannelFirst, EnsureType, ScaleIntensityRange

from .train_3d_monai import CLASSES, CLASS2IDX, build_case_day_slices, parse_scan_filename


def rle_encode(img: np.ndarray) -> str:
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


class TestVolumeDataset(Dataset):
    def __init__(
        self,
        case_day_slices: Dict[str, List[str]],
        max_cases: Optional[int] = None,
        max_slices: Optional[int] = None,
    ):
        self.case_days = sorted(case_day_slices.keys())
        if max_cases is not None:
            self.case_days = self.case_days[:max_cases]
        self.case_day_slices = case_day_slices
        self.max_slices = max_slices
        self.xform = Compose([
            EnsureChannelFirst(channel_dim="no_channel"),
            ScaleIntensityRange(a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
            EnsureType(data_type="tensor"),
        ])

    def __len__(self):
        return len(self.case_days)

    def __getitem__(self, idx: int):
        case_day = self.case_days[idx]
        slice_files = self.case_day_slices[case_day]
        if self.max_slices is not None:
            slice_files = slice_files[: self.max_slices]
        _, H, W = parse_scan_filename(slice_files[0])
        D = len(slice_files)

        img_vol = np.zeros((D, H, W), dtype=np.uint8)
        slice_indices = []
        for z, f in enumerate(slice_files):
            slice_idx, h, w = parse_scan_filename(f)
            if h != H or w != W:
                raise ValueError(f"Inconsistent shape in {case_day}: {f}")
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f"Failed to read image: {f}")
            img_vol[z] = img
            slice_indices.append(slice_idx)

        x = self.xform(img_vol)  # (1,D,H,W)
        return x, case_day, slice_indices


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

    data_root = args.data_root or "./input/uw-madison-gi-tract-image-segmentation"
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

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=0.2,
    )

    ckpt = torch.load(weights, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    roi_size = (
        args.sw_patch_d or 80,
        args.sw_patch_h or 224,
        args.sw_patch_w or 224,
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
