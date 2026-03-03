from monai.transforms import MapTransform
import cv2
import numpy as np
import os

from .data_utils import parse_scan_filename
from .rle import rle_decode
from .constants import CLASSES, CLASS2IDX


class LoadCaseDayVolumed(MapTransform):
    def __init__(self, keys, case_day_slices, rle_index):
        super().__init__(keys)
        self.case_day_slices = case_day_slices
        self.rle_index = rle_index

    @staticmethod
    def _read_scan(path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Failed to read scan: {path}")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
        slice_idxs: list[int] = []

        for z, f in enumerate(slice_files):
            slice_idx = parse_scan_filename(os.path.basename(f))[1]
            slice_idxs.append(int(slice_idx))
            img = self._read_scan(f)
            img_vol[z] = img

            rle_map = self.rle_index.get((case_day, slice_idx), {})
            for cls_name, cls_i in CLASS2IDX.items():
                m2d = rle_decode(rle_map.get(cls_name, ""), H, W)
                mask_vol[cls_i, z] = m2d

        d["image"] = img_vol
        d["label"] = mask_vol
        d["orig_shape"] = (int(D), int(H), int(W))
        d["slice_idxs"] = slice_idxs
        return d


class LoadCaseDayImaged(MapTransform):
    def __init__(self, keys, case_day_slices):
        super().__init__(keys)
        self.case_day_slices = case_day_slices

    @staticmethod
    def _read_scan(path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Failed to read scan: {path}")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def __call__(self, data):
        d = dict(data)
        case_day = d["case_day"]
        slice_files = self.case_day_slices[case_day]

        img0 = self._read_scan(slice_files[0])
        H, W = img0.shape
        D = len(slice_files)

        img_vol = np.zeros((D, H, W), dtype=img0.dtype)
        slice_idxs: list[int] = []
        for z, f in enumerate(slice_files):
            slice_idx = parse_scan_filename(os.path.basename(f))[1]
            slice_idxs.append(int(slice_idx))
            img = self._read_scan(f)
            img_vol[z] = img

        d["image"] = img_vol
        d["orig_shape"] = (int(D), int(H), int(W))
        d["slice_idxs"] = slice_idxs
        return d
    
