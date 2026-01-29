#!/bin/bash
set -euo pipefail

# Downloads the MONAI SwinUNETR BTCV pretrained weights (model.pt) from Hugging Face.
# Output: pretrained/swin_unetr_btcv/models/model.pt

OUT_DIR="pretrained/swin_unetr_btcv/models"
OUT_PATH="${OUT_DIR}/model.pt"

mkdir -p "$OUT_DIR"

URL="https://huggingface.co/MONAI/swin_unetr_btcv_segmentation/resolve/main/models/model.pt"

if command -v curl >/dev/null 2>&1; then
  curl -L --fail --retry 3 -o "$OUT_PATH" "$URL"
elif command -v wget >/dev/null 2>&1; then
  wget -O "$OUT_PATH" "$URL"
else
  echo "Need curl or wget to download weights." >&2
  exit 1
fi

echo "Saved: $OUT_PATH"

