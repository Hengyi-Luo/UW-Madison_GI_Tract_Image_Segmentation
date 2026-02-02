from __future__ import annotations

from typing import Tuple

import torch


def _require_multiple_of(v: int, m: int, name: str) -> None:
    if v % m != 0:
        raise ValueError(f"{name}={v} must be divisible by {m} (got {v}).")


def validate_patch_size(model_name: str, patch_size: Tuple[int, int, int]) -> None:
    if model_name.lower() in {"swin_unetr", "swinunetr"}:
        # SwinUNETR downsamples 5 times (2**5=32).
        for n, v in zip(("patch_d", "patch_h", "patch_w"), patch_size):
            _require_multiple_of(int(v), 32, n)


def build_model(
    model_name: str,
    patch_size: Tuple[int, int, int],
    in_channels: int = 1,
    out_channels: int = 3,
    feature_size: int = 48,
    use_checkpoint: bool = False,
) -> torch.nn.Module:
    model_name = (model_name or "unet").lower()
    validate_patch_size(model_name, patch_size)

    if model_name == "unet":
        from monai.networks.nets import UNet

        return UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            dropout=0.2,
        )

    if model_name in {"swin_unetr", "swinunetr"}:
        from monai.networks.nets import SwinUNETR

        # MONAI SwinUNETR signatures differ by version:
        # - Some versions require `img_size` (tuple) in the constructor.
        # - Other versions don't have `img_size` and use a `patch_size` token size (default=2),
        #   with a runtime check that input spatial dims are divisible by patch_size**5 (usually 32).
        import inspect

        sig = inspect.signature(SwinUNETR.__init__)
        if "img_size" in sig.parameters:
            img_param = sig.parameters["img_size"]
            if img_param.kind == inspect.Parameter.POSITIONAL_ONLY:
                return SwinUNETR(
                    patch_size,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    feature_size=int(feature_size),
                    use_checkpoint=bool(use_checkpoint),
                )
            return SwinUNETR(
                img_size=patch_size,
                in_channels=in_channels,
                out_channels=out_channels,
                feature_size=int(feature_size),
                use_checkpoint=bool(use_checkpoint),
            )

        # No `img_size` parameter: DO NOT pass our ROI size as `patch_size`.
        return SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=int(feature_size),
            use_checkpoint=bool(use_checkpoint),
            spatial_dims=3,
        )

    raise ValueError(f"Unknown model: {model_name}. Supported: unet, swin_unetr")
