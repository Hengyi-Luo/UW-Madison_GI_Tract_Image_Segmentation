from __future__ import annotations

import os
import random
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    EnsureType,
    RandFlip,
    RandRotate90,
    ScaleIntensityRange,
)

from .constants import CLASSES, CLASS2IDX
from .data_utils import parse_scan_filename
from .rle import rle_decode


class UWGI3DPatchDataset(Dataset):
    """Loads a case_day volume (D,H,W) + masks (3,D,H,W), then samples random 3D patches.

    Always returns a fixed patch shape (patch_d, patch_h, patch_w) via crop+pad.
    """

    def __init__(
        self,
        case_days: List[str],
        case_day_slices: Dict[str, List[str]],
        rle_index: Dict[Tuple[str, int], Dict[str, str]],
        patch_size: Tuple[int, int, int],  # (D,H,W)
        samples_per_volume: int,
        cache_in_ram: bool = False,
        seed: int = 42,
        is_train: bool = True,
        verbose_shape_fix: bool = False,
    ):
        self.case_days = case_days
        self.case_day_slices = case_day_slices
        self.rle_index = rle_index

        self.patch_d, self.patch_h, self.patch_w = patch_size
        self.samples_per_volume = samples_per_volume
        self.cache_in_ram = cache_in_ram
        self.is_train = is_train
        self.verbose_shape_fix = verbose_shape_fix

        self.rng = random.Random(seed)
        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        self.xform = Compose(
            [
                EnsureChannelFirst(channel_dim="no_channel"),  # (D,H,W) -> (1,D,H,W)
                ScaleIntensityRange(a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
                EnsureType(data_type="tensor"),
            ]
        )

        self.aug = (
            Compose(
                [
                    RandFlip(prob=0.5, spatial_axis=2),  # flip W
                    RandFlip(prob=0.5, spatial_axis=1),  # flip H
                    RandRotate90(prob=0.3, max_k=3, spatial_axes=(1, 2)),  # rotate in H-W plane
                ]
            )
            if is_train
            else None
        )

        self._index = [cd for cd in self.case_days for _ in range(self.samples_per_volume)]

    def __len__(self):
        return len(self._index)

    @staticmethod
    def _fix_to_hw(img2d: np.ndarray, H: int, W: int, mode: str) -> np.ndarray:
        if img2d.shape == (H, W):
            return img2d
        if img2d.shape == (W, H):
            return img2d.T
        interp = cv2.INTER_AREA if mode == "image" else cv2.INTER_NEAREST
        return cv2.resize(img2d, (W, H), interpolation=interp)

    def _load_volume(self, case_day: str) -> Tuple[np.ndarray, np.ndarray]:
        if self.cache_in_ram and case_day in self._cache:
            return self._cache[case_day]

        slice_files = self.case_day_slices[case_day]
        if len(slice_files) == 0:
            raise RuntimeError(f"No slice files for {case_day}")

        img0 = cv2.imread(slice_files[0], cv2.IMREAD_GRAYSCALE)
        if img0 is None:
            raise RuntimeError(f"Failed to read image: {slice_files[0]}")
        H, W = img0.shape
        D = len(slice_files)

        img_vol = np.zeros((D, H, W), dtype=np.uint8)
        mask_vol = np.zeros((len(CLASSES), D, H, W), dtype=np.uint8)

        for z, f in enumerate(slice_files):
            slice_idx, _, _ = parse_scan_filename(f)

            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f"Failed to read image: {f}")

            if img.shape != (H, W) and self.verbose_shape_fix:
                print(
                    f"[WARN] shape mismatch {case_day} {os.path.basename(f)} "
                    f"img.shape={img.shape} expected={(H, W)}"
                )

            img = self._fix_to_hw(img, H, W, mode="image")
            img_vol[z] = img

            rle_map = self.rle_index.get((case_day, slice_idx), {})
            for cls_name, cls_i in CLASS2IDX.items():
                m2d = rle_decode(rle_map.get(cls_name, ""), H, W)
                if m2d.shape != (H, W):
                    m2d = self._fix_to_hw(m2d, H, W, mode="mask")
                mask_vol[cls_i, z] = m2d

        if self.cache_in_ram:
            self._cache[case_day] = (img_vol, mask_vol)

        return img_vol, mask_vol

    @staticmethod
    def _pad_3d(vol: np.ndarray, target: Tuple[int, int, int], pad_value: int = 0) -> np.ndarray:
        td, th, tw = target
        d, h, w = vol.shape
        pd, ph, pw = max(0, td - d), max(0, th - h), max(0, tw - w)
        if pd == 0 and ph == 0 and pw == 0:
            return vol
        return np.pad(vol, ((0, pd), (0, ph), (0, pw)), mode="constant", constant_values=pad_value)

    @staticmethod
    def _pad_4d(vol: np.ndarray, target: Tuple[int, int, int], pad_value: int = 0) -> np.ndarray:
        td, th, tw = target
        _, d, h, w = vol.shape
        pd, ph, pw = max(0, td - d), max(0, th - h), max(0, tw - w)
        if pd == 0 and ph == 0 and pw == 0:
            return vol
        return np.pad(vol, ((0, 0), (0, pd), (0, ph), (0, pw)), mode="constant", constant_values=pad_value)

    def __getitem__(self, idx: int):
        case_day = self._index[idx]
        img_vol, mask_vol = self._load_volume(case_day)
        D, H, W = img_vol.shape

        pd, ph, pw = self.patch_d, self.patch_h, self.patch_w
        z0 = 0 if D <= pd else self.rng.randint(0, D - pd)
        y0 = 0 if H <= ph else self.rng.randint(0, H - ph)
        x0 = 0 if W <= pw else self.rng.randint(0, W - pw)

        z1, y1, x1 = min(D, z0 + pd), min(H, y0 + ph), min(W, x0 + pw)
        img_patch = img_vol[z0:z1, y0:y1, x0:x1]
        mask_patch = mask_vol[:, z0:z1, y0:y1, x0:x1]

        img_patch = self._pad_3d(img_patch, (pd, ph, pw), pad_value=0)
        mask_patch = self._pad_4d(mask_patch, (pd, ph, pw), pad_value=0)

        x = self.xform(img_patch)
        y = torch.from_numpy(mask_patch.astype(np.float32))

        if self.aug is not None:
            xy = torch.cat([x, y], dim=0)  # (4,D,H,W)
            xy = self.aug(xy)
            x = xy[0:1]
            y = xy[1:]

        return x, y, case_day


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
        self.xform = Compose(
            [
                EnsureChannelFirst(channel_dim="no_channel"),
                ScaleIntensityRange(a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
                EnsureType(data_type="tensor"),
            ]
        )

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

        x = self.xform(img_vol)
        return x, case_day, slice_indices

