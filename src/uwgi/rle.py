from __future__ import annotations

import numpy as np


def rle_decode(rle: str, height: int, width: int) -> np.ndarray:
    """Kaggle RLE decode for 2D mask.

    Note: UW-Madison GI competition RLE is column-major order.
    """
    if rle is None or rle == "" or (isinstance(rle, float) and np.isnan(rle)):
        return np.zeros((height, width), dtype=np.uint8)

    s = rle.strip().split()
    starts = np.asarray(s[0::2], dtype=int) - 1
    lengths = np.asarray(s[1::2], dtype=int)
    ends = starts + lengths

    img = np.zeros(height * width, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    # column-major to (H,W)
    return img.reshape((width, height)).T


def rle_encode(img: np.ndarray) -> str:
    """Kaggle RLE encode for 2D mask (expects {0,1} uint8)."""
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)

