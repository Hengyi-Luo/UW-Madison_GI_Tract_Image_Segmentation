#!/bin/bash
# run_eval.sh
# Usage:
#   bash scripts/run_eval.sh configs/eval.yaml

set -e

CONFIG_FILE="${1:-configs/eval.yaml}"
EXTRA_ARGS="${@:2}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "=== UW-Madison GI Evaluation (Dice) ==="
echo "Config: $CONFIG_FILE"
echo ""

ARGS=$(python3 -c "
import sys
try:
    import yaml
except Exception:
    print('PyYAML not installed. Please: pip install pyyaml', file=sys.stderr)
    raise

with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f) or {}

args = []
for k, v in config.items():
    if v is None or v == 'null':
        continue
    if isinstance(v, bool):
        if v:
            args.append(f'--{k}')
    else:
        args.append(f'--{k}')
        args.append(str(v))

print(' '.join(args))
")

echo "Arguments: $ARGS"
echo ""

PYTHONPATH=src python -m uwgi.eval_cli --config "$CONFIG_FILE" $ARGS $EXTRA_ARGS

