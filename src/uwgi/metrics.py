from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def dice_per_class_2d_kaggle(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Kaggle-style Dice per class for a single 2D slice.

    pred/target: (C,H,W) in {0,1} (or boolean/float).
    Dice is defined as 0 when both pred and target are empty for that class.
    Returns: (C,) float32.
    """
    pred = np.asarray(pred).astype(np.float32, copy=False)
    target = np.asarray(target).astype(np.float32, copy=False)
    if pred.shape != target.shape or pred.ndim != 3:
        raise ValueError(f"Expected pred/target with shape (C,H,W) and same shape, got {pred.shape} {target.shape}")

    inter = (pred * target).sum(axis=(-2, -1))
    pred_sum = pred.sum(axis=(-2, -1))
    tgt_sum = target.sum(axis=(-2, -1))
    denom = pred_sum + tgt_sum
    dice = np.where(denom > 0, (2.0 * inter + eps) / (denom + eps), 0.0)
    return dice.astype(np.float32)


def hausdorff_distance_3d_kaggle(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    eps: float = 1e-8,
    return_per_class: bool = True,
) -> Tuple[np.ndarray, float] | float:
    """
    Kaggle-style 3D Hausdorff distance (bounded 0..1) for a 3D volume.

    pred/target: (C,D,H,W) in {0,1}.
    - If both pred and target are empty => distance=0
    - If only one is empty => distance=1
    - Otherwise Hausdorff distance is computed in voxel space (z depth=1),
      then normalized by the volume diagonal so the result is in [0,1].

    Returns:
      - if return_per_class: (per_class: (C,), mean: float)
      - else: mean: float
    """
    try:
        from scipy.ndimage import distance_transform_edt
    except Exception as e:  # pragma: no cover
        raise ImportError("scipy is required for hausdorff_distance_3d_kaggle (scipy.ndimage.distance_transform_edt)") from e

    pred = np.asarray(pred)
    target = np.asarray(target)
    if pred.shape != target.shape or pred.ndim != 4:
        raise ValueError(f"Expected pred/target with shape (C,D,H,W) and same shape, got {pred.shape} {target.shape}")

    c, d, h, w = pred.shape
    diag = float(np.sqrt(max(1, d - 1) ** 2 + max(1, h - 1) ** 2 + max(1, w - 1) ** 2) + eps)

    out = np.zeros((c,), dtype=np.float32)
    for i in range(c):
        a = pred[i].astype(bool, copy=False)
        b = target[i].astype(bool, copy=False)
        a_any = bool(a.any())
        b_any = bool(b.any())
        if not a_any and not b_any:
            out[i] = 0.0
            continue
        if a_any != b_any:
            out[i] = 1.0
            continue

        # Directed HD: max_{p in A} min_{q in B} ||p-q||, via distance transform.
        dt_b = distance_transform_edt(~b)  # distance to nearest True in b
        d_ab = float(dt_b[a].max(initial=0.0))
        dt_a = distance_transform_edt(~a)
        d_ba = float(dt_a[b].max(initial=0.0))
        hd = max(d_ab, d_ba)
        out[i] = float(hd / diag)

    mean = float(out.mean()) if out.size else 0.0
    return (out, mean) if return_per_class else mean


def hausdorff_score_3d_scipy_kaggle(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    empty_score: float = 0.0,
    return_per_class: bool = True,
) -> Tuple[np.ndarray, float] | float:
    """
    Kaggle-style 3D Hausdorff *score* using SciPy's Hausdorff implementation.

    Steps (per class):
      1) Build 3D point clouds from foreground voxels using coordinates (z,y,x).
      2) Normalize coordinates by image/volume size to put points in a unit box.
      3) Compute symmetric Hausdorff distance with scipy.spatial.distance.directed_hausdorff.
      4) Normalize by sqrt(3) to bound to [0,1], then convert to score via (1 - distance).

    Empty handling (per class):
      - if both pred and target are empty: score = empty_score
      - if only one is empty: score = 0

    pred/target: (C,D,H,W) in {0,1}.
    Returns:
      - if return_per_class: (per_class_scores: (C,), mean_score: float)
      - else: mean_score: float
    """
    try:
        from scipy.spatial.distance import directed_hausdorff
    except Exception as e:  # pragma: no cover
        raise ImportError("scipy is required for hausdorff_score_3d_scipy_kaggle (scipy.spatial.distance.directed_hausdorff)") from e

    pred = np.asarray(pred)
    target = np.asarray(target)
    if pred.shape != target.shape or pred.ndim != 4:
        raise ValueError(f"Expected pred/target with shape (C,D,H,W) and same shape, got {pred.shape} {target.shape}")

    c, d, h, w = pred.shape
    denom = np.asarray([max(1, d - 1), max(1, h - 1), max(1, w - 1)], dtype=np.float32)
    max_dist = float(np.sqrt(3.0))

    out = np.zeros((c,), dtype=np.float32)
    for i in range(c):
        a = np.asarray(pred[i]).astype(bool, copy=False)
        b = np.asarray(target[i]).astype(bool, copy=False)

        a_any = bool(a.any())
        b_any = bool(b.any())
        if not a_any and not b_any:
            out[i] = float(empty_score)
            continue
        if a_any != b_any:
            out[i] = 0.0
            continue

        pts_a = np.argwhere(a).astype(np.float32, copy=False) / denom  # (N,3) in z,y,x
        pts_b = np.argwhere(b).astype(np.float32, copy=False) / denom

        # scipy returns directed Hausdorff distance; make it symmetric.
        d_ab = float(directed_hausdorff(pts_a, pts_b)[0])
        d_ba = float(directed_hausdorff(pts_b, pts_a)[0])
        hd = max(d_ab, d_ba)
        hd_norm = float(min(1.0, hd / max_dist))
        out[i] = float(1.0 - hd_norm)

    mean = float(out.mean()) if out.size else 0.0
    return (out, mean) if return_per_class else mean
