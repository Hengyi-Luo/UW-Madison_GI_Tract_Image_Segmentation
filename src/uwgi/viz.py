from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np

from .constants import CLASSES
from .data_utils import parse_scan_filename
from .rle import rle_decode


def load_gt_masks_for_slice(
    case_day: str,
    slice_path: str,
    rle_index: Dict[Tuple[str, int], Dict[str, str]],
    classes: Iterable[str] = CLASSES,
) -> Dict[str, np.ndarray]:
    """
    Returns {class_name: mask2d_uint8} for one slice, using train.csv RLEs.
    `slice_path` must point to a PNG under train/.../scans/.
    """
    img = cv2.imread(slice_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {slice_path}")
    h, w = img.shape
    slice_idx, _, _ = parse_scan_filename(slice_path)

    rle_map = rle_index.get((case_day, int(slice_idx)), {}) or {}
    masks: Dict[str, np.ndarray] = {}
    for cls in classes:
        masks[str(cls)] = rle_decode(rle_map.get(str(cls), "") or "", int(h), int(w))
    return masks


def load_model_from_checkpoint(
    weights: str,
    *,
    device: Optional[str] = None,
    model_override: Optional[str] = None,
):
    """
    Loads a model + checkpoint cfg from a training checkpoint produced by `train_3d_monai.py`.

    Returns (model, torch_device, cfg_dict, patch_size_tuple).
    """
    import torch

    from .models import build_model

    ckpt = torch.load(weights, map_location="cpu")
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError(f"Unrecognized checkpoint format: {weights}")

    cfg = ckpt.get("cfg", {}) if isinstance(ckpt.get("cfg", {}), dict) else {}
    model_name = (model_override or cfg.get("model", "unet") or "unet").lower()
    patch_size = (
        int(cfg.get("patch_d", 96)),
        int(cfg.get("patch_h", 224)),
        int(cfg.get("patch_w", 224)),
    )

    model = build_model(
        model_name=model_name,
        patch_size=patch_size,
        in_channels=1,
        out_channels=len(CLASSES),
        feature_size=int(cfg.get("feature_size", 48)),
        use_checkpoint=bool(cfg.get("use_checkpoint", False)),
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    if device is None:
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch_device = torch.device(device)
    model.to(torch_device)

    return model, torch_device, cfg, patch_size


def predict_case_day_volume(
    case_day: str,
    case_day_slices: Dict[str, list[str]],
    rle_index: Dict[Tuple[str, int], Dict[str, str]],
    model,
    *,
    device=None,
    mode: str = "sw",
    roi_size: Optional[Tuple[int, int, int]] = None,
    sw_batch_size: int = 1,
    overlap: float = 0.25,
    threshold: float = 0.5,
    spatial_order: str = "DHW",
    normalize: str = "none",
    tta: bool = False,
    tta_dims: Optional[Iterable[Iterable[int]]] = None,
):
    """
    Returns (pred_mask_uint8, gt_mask_uint8, x_tensor_cpu) where:
      - pred_mask_uint8: (C,D,H,W) in {0,1}
      - gt_mask_uint8:   (C,D,H,W) in {0,1}
      - x_tensor_cpu:    (1,D,H,W) float32 in [0,1] (dataset-transformed input)
    """
    import torch

    from monai.inferers import sliding_window_inference

    from .datasets import UWGI3DFullVolumeDataset

    if case_day not in case_day_slices:
        raise KeyError(f"case_day not found in case_day_slices: {case_day}")

    ds = UWGI3DFullVolumeDataset(
        case_days=[case_day],
        case_day_slices=case_day_slices,
        rle_index=rle_index,
        cache_in_ram=False,
        verbose_shape_fix=False,
    )
    x, y, _ = ds[0]  # x:(1,D,H,W) y:(C,D,H,W)
    x_b = x.unsqueeze(0)

    if device is None:
        device = next(model.parameters()).device

    spatial_order = (spatial_order or "DHW").upper()
    if spatial_order not in {"DHW", "WHD"}:
        raise ValueError(f"Unsupported spatial_order={spatial_order}. Supported: DHW, WHD")

    normalize = (normalize or "none").lower()
    if normalize not in {"none", "volume_max"}:
        raise ValueError(f"Unsupported normalize={normalize}. Supported: none, volume_max")

    mode = (mode or "sw").lower()
    with torch.no_grad():
        x_in = x_b
        if spatial_order == "WHD":
            # (B,C,D,H,W) -> (B,C,W,H,D)
            x_in = x_in.permute(0, 1, 4, 3, 2)

        if normalize == "volume_max":
            # Match kernels that normalize each 3D sample by its own max value.
            mx = x_in.amax(dim=(2, 3, 4), keepdim=True).clamp_min(1e-6)
            x_in = x_in / mx

        def _infer(inp: torch.Tensor) -> torch.Tensor:
            if mode == "direct":
                return model(inp.to(device))
            if mode == "sw":
                if roi_size is None:
                    raise ValueError("roi_size is required for mode='sw'")
                return sliding_window_inference(
                    inp.to(device),
                    roi_size,
                    int(sw_batch_size),
                    model,
                    overlap=float(overlap),
                )
            raise ValueError(f"Unknown mode: {mode}. Supported: sw, direct")

        if tta_dims is None:
            if spatial_order == "WHD":
                # Input is (B,C,W,H,D): flip W/H like the reference notebook.
                dims_list = [(), (2,), (3,), (2, 3)]
            else:
                # Input is (B,C,D,H,W): flip H/W.
                dims_list = [(), (3,), (4,), (3, 4)]
        else:
            dims_list = [tuple(int(d) for d in dims) for dims in tta_dims]
            if () not in dims_list:
                dims_list = [()] + dims_list

        if not bool(tta):
            dims_list = [()]

        logits_sum = None
        for dims in dims_list:
            x_f = torch.flip(x_in, dims=dims) if dims else x_in
            l = _infer(x_f)
            if dims:
                l = torch.flip(l, dims=dims)
            logits_sum = l if logits_sum is None else (logits_sum + l)
        logits = logits_sum / float(len(dims_list))

        if spatial_order == "WHD":
            # (B,C,W,H,D) -> (B,C,D,H,W)
            logits = logits.permute(0, 1, 4, 3, 2)

        prob = torch.sigmoid(logits)
        pred = (prob > float(threshold)).to(torch.uint8)[0].cpu()  # (C,D,H,W)

    gt = (y > 0.5).to(torch.uint8).cpu()
    return pred.numpy(), gt.numpy(), x.cpu()


def plot_case_day_slice_gt(
    case_day: str,
    slice_pos: int,
    case_day_slices: Dict[str, list[str]],
    rle_index: Dict[Tuple[str, int], Dict[str, str]],
    *,
    classes: Iterable[str] = CLASSES,
    overlay: bool = True,
    figsize: Tuple[int, int] = (12, 8),
    overlay_figsize: Tuple[int, int] = (6, 6),
    alpha: float = 0.4,
):
    """
    Notebook-friendly visualization similar to your example:
    - a 2x2 grid (image + 3 class masks)
    - optional overlay plot with 3 masks

    Parameters
    - slice_pos: index into case_day_slices[case_day] (not the Kaggle slice_idx).
      slice_idx will be derived from the filename for correct RLE lookup.
    """
    import matplotlib.pyplot as plt

    slice_paths = case_day_slices[case_day]
    if slice_pos < 0 or slice_pos >= len(slice_paths):
        raise IndexError(f"slice_pos={slice_pos} out of range (0..{len(slice_paths)-1}) for {case_day}")

    slice_path = slice_paths[slice_pos]
    img = cv2.imread(slice_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {slice_path}")

    cls_list = list(classes)
    if len(cls_list) != 3:
        raise ValueError(f"Expected 3 classes for plotting, got {len(cls_list)}: {cls_list}")

    masks = load_gt_masks_for_slice(case_day, slice_path, rle_index, classes=cls_list)

    fig1 = plt.figure(figsize=figsize)
    plt.subplot(2, 2, 1)
    plt.title("image")
    plt.imshow(img, cmap="gray")

    cmaps = ["Reds", "Greens", "Blues"]
    for i, cls in enumerate(cls_list):
        plt.subplot(2, 2, 2 + i)
        plt.title(str(cls))
        plt.imshow(masks[str(cls)], cmap=cmaps[i])

    plt.tight_layout()

    fig2 = None
    if overlay:
        fig2 = plt.figure(figsize=overlay_figsize)
        plt.title("overlay (C order)")
        plt.imshow(img, cmap="gray")
        for i, cls in enumerate(cls_list):
            plt.imshow(masks[str(cls)], alpha=float(alpha), cmap=cmaps[i])
        plt.tight_layout()

    return fig1, fig2, slice_path


def plot_case_day_slice_gt_pred(
    case_day: str,
    slice_pos: int,
    case_day_slices: Dict[str, list[str]],
    rle_index: Dict[Tuple[str, int], Dict[str, str]],
    *,
    model=None,
    weights: Optional[str] = None,
    device: Optional[str] = None,
    model_override: Optional[str] = None,
    pred_vol: Optional[np.ndarray] = None,
    gt_vol: Optional[np.ndarray] = None,
    mode: str = "sw",
    roi_size: Optional[Tuple[int, int, int]] = None,
    sw_batch_size: int = 1,
    overlap: float = 0.25,
    threshold: float = 0.5,
    spatial_order: str = "DHW",
    classes: Iterable[str] = CLASSES,
    figsize: Tuple[int, int] = (14, 7),
    overlay_figsize: Tuple[int, int] = (12, 6),
    alpha: float = 0.4,
):
    """
    Plots GT vs prediction for a single slice.

    You can either pass a ready `model=...` (already on GPU/CPU), or pass `weights=...`
    to auto-load a model from a training checkpoint.

    Returns (fig_grid, fig_overlay, slice_path, pred_mask2d_by_class, gt_mask2d_by_class).
    """
    import matplotlib.pyplot as plt

    cls_list = list(classes)
    if len(cls_list) != 3:
        raise ValueError(f"Expected 3 classes for plotting, got {len(cls_list)}: {cls_list}")

    have_vols = (pred_vol is not None) and (gt_vol is not None)
    if not have_vols:
        if model is None:
            if not weights:
                raise ValueError(
                    "Provide either (pred_vol, gt_vol), or model=..., or weights=... for prediction plotting."
                )
            model, _, _, patch_size = load_model_from_checkpoint(weights, device=device, model_override=model_override)
            if roi_size is None and (mode or "sw").lower() == "sw":
                roi_size = patch_size

    slice_paths = case_day_slices[case_day]
    if slice_pos < 0 or slice_pos >= len(slice_paths):
        raise IndexError(f"slice_pos={slice_pos} out of range (0..{len(slice_paths)-1}) for {case_day}")
    slice_path = slice_paths[slice_pos]

    img = cv2.imread(slice_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {slice_path}")

    if not have_vols:
        pred_vol, gt_vol, _ = predict_case_day_volume(
            case_day,
            case_day_slices,
            rle_index,
            model,
            mode=mode,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            overlap=overlap,
            threshold=threshold,
            spatial_order=spatial_order,
        )
    else:
        pred_vol = np.asarray(pred_vol)
        gt_vol = np.asarray(gt_vol)
        if pred_vol.ndim != 4 or gt_vol.ndim != 4:
            raise ValueError(f"Expected pred_vol/gt_vol with shape (C,D,H,W); got {pred_vol.shape} {gt_vol.shape}")
        if pred_vol.shape != gt_vol.shape:
            raise ValueError(f"pred_vol and gt_vol must have the same shape; got {pred_vol.shape} vs {gt_vol.shape}")
        if slice_pos < 0 or slice_pos >= int(pred_vol.shape[1]):
            raise IndexError(f"slice_pos={slice_pos} out of range (0..{int(pred_vol.shape[1])-1}) for pred_vol")

    pred2d = {cls_list[i]: pred_vol[i, slice_pos] for i in range(3)}
    gt2d = {cls_list[i]: gt_vol[i, slice_pos] for i in range(3)}

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

    return fig_grid, fig_overlay, slice_path, pred2d, gt2d
