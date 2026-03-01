from monai.transforms import MapTransform
import cv2
import numpy as np
import os

from .data_utils import parse_scan_filename
from .rle import rle_decode
from .constants import CLASSES, CLASS2IDX


class LoadCaseDayVolumed(MapTransform):
    def __init__(self, keys, case_day_slices, rle_index, scan_read_mode: str = "unchanged"):
        super().__init__(keys)
        self.case_day_slices = case_day_slices
        self.rle_index = rle_index
        self.scan_read_mode = str(scan_read_mode)

    def _read_scan(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Failed to read scan: {path}")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.scan_read_mode == "legacy_uint8":
            if img.dtype == np.uint16:
                img = (img >> 8).astype(np.uint8, copy=False)
            elif img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8, copy=False)
        elif self.scan_read_mode != "unchanged":
            raise ValueError(
                f"Unknown scan_read_mode={self.scan_read_mode!r}; expected 'unchanged' or 'legacy_uint8'."
            )
        return img

    def __call__(self, data):
        d = dict(data)
        case_day = d["case_day"]
        slice_files = self.case_day_slices[case_day]

        img0 = self._read_scan(slice_files[0])
        H, W = img0.shape
        D = len(slice_files)

        img_vol = np.zeros((D, H, W), dtype=img0.dtype)
        mask_vol = np.zeros((len(CLASSES), D, H, W), dtype=np.uint8)

        for z, f in enumerate(slice_files):
            slice_idx = parse_scan_filename(os.path.basename(f))[1]
            img = self._read_scan(f)
            img_vol[z] = img

            rle_map = self.rle_index.get((case_day, slice_idx), {})
            for cls_name, cls_i in CLASS2IDX.items():
                m2d = rle_decode(rle_map.get(cls_name, ""), H, W)
                mask_vol[cls_i, z] = m2d

        d["image"] = img_vol
        d["label"] = mask_vol
        return d
    
