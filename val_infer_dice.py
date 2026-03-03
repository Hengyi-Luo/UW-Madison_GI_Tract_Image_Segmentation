#!/usr/bin/env python3
"""
Backward-compatible entrypoint.

This repo's inference script is now `scripts/infer_unet3d.py`.
"""

from __future__ import annotations

from scripts.infer_unet3d import main


if __name__ == "__main__":
    main()

