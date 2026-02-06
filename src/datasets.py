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

    def __call__(self, data):
        d = dict(data)
        case_day = d["case_day"]
        slice_files = self.case_day_slices[case_day]

        img0 = cv2.imread(slice_files[0], cv2.IMREAD_GRAYSCALE)
        H, W = img0.shape
        D = len(slice_files)

        img_vol = np.zeros((D, H, W), dtype=np.uint8)
        mask_vol = np.zeros((len(CLASSES), D, H, W), dtype=np.uint8)

        for z, f in enumerate(slice_files):
            slice_idx = parse_scan_filename(os.path.basename(f))[1]
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            img_vol[z] = img

            rle_map = self.rle_index.get((case_day, slice_idx), {})
            for cls_name, cls_i in CLASS2IDX.items():
                m2d = rle_decode(rle_map.get(cls_name, ""), H, W)
                mask_vol[cls_i, z] = m2d

        d["image"] = img_vol
        d["label"] = mask_vol
        return d
    
