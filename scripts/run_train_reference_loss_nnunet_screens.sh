#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/jeremiah/github/UW-Madison_GI_Tract_Image_Segmentation"
PYTHON_BIN="/home/jeremiah/miniconda/envs/uwgi/bin/python"
LOG_DIR="$REPO_ROOT/outputs/screen_logs"

SESSION_1="train_safe_nnunet"
SESSION_2="train_nnunet_like"
SESSION_SEQ="train_ref_nnunet_seq"

CONFIG_1="configs/train_unet3d_reference_loss_safe_aug_nnunet_style.py"
CONFIG_2="configs/train_unet3d_reference_loss_nnunet_like.py"

LOG_1="$LOG_DIR/train_unet3d_reference_loss_safe_aug_nnunet_style.log"
LOG_2="$LOG_DIR/train_unet3d_reference_loss_nnunet_like.log"

mkdir -p "$LOG_DIR"

if screen -list | grep -q "[.]$SESSION_1[[:space:]]"; then
  echo "legacy screen session already exists: $SESSION_1" >&2
  exit 1
fi

if screen -list | grep -q "[.]$SESSION_2[[:space:]]"; then
  echo "legacy screen session already exists: $SESSION_2" >&2
  exit 1
fi

if screen -list | grep -q "[.]$SESSION_SEQ[[:space:]]"; then
  echo "screen session already exists: $SESSION_SEQ" >&2
  exit 1
fi

screen -dmS "$SESSION_SEQ" bash -lc "
cd '$REPO_ROOT' && \
'$PYTHON_BIN' -u scripts/train_unet3d.py --config '$CONFIG_1' 2>&1 | tee '$LOG_1' && \
'$PYTHON_BIN' -u scripts/train_unet3d.py --config '$CONFIG_2' 2>&1 | tee '$LOG_2'
"

echo "Started sequential screen session:"
echo "  $SESSION_SEQ"
echo "Execution order:"
echo "  1. $CONFIG_1"
echo "  2. $CONFIG_2"
echo
echo "Logs:"
echo "  $LOG_1"
echo "  $LOG_2"
echo
screen -ls
