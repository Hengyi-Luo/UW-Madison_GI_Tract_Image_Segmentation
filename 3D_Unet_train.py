#!/usr/bin/env python3
"""
Legacy entrypoint kept for convenience.

Use:
  python 3D_Unet_train.py --config configs/train_unet3d.yaml
"""

from scripts.train_unet3d import main


if __name__ == "__main__":
    main()

