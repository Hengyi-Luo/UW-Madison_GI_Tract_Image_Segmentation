#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/train_unet3d.yaml}"
shift || true

python scripts/train_unet3d.py --config "$CONFIG" "$@"

