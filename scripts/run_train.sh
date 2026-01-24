#!/bin/bash
# run_train.sh
# Usage:
#   bash scripts/run_train.sh configs/train.yaml

set -e

CONFIG_FILE="${1:-configs/train.yaml}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "=== UW-Madison GI Training ==="
echo "Config: $CONFIG_FILE"
echo ""

ARGS=$(python3 -c "
import sys
try:
    import yaml
except Exception as e:
    print('PyYAML not installed. Please: pip install pyyaml', file=sys.stderr)
    raise

with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)

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

python train_cli.py $ARGS
