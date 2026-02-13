#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
  echo "Usage: $0 <CONFIG_YAML> [OUTPUT_BASE]" >&2
  exit 1
fi

CONFIG_YAML="$1"
OUTPUT_BASE="${2:-outputs/grpo}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

if [ ! -f "$CONFIG_YAML" ]; then
  echo "CONFIG_YAML not found: $CONFIG_YAML" >&2
  exit 1
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="$OUTPUT_BASE/$TIMESTAMP"
mkdir -p "$RUN_DIR"

cp "$CONFIG_YAML" "$RUN_DIR/config_snapshot.yaml"

LOG_FILE="$RUN_DIR/train.log"
PID_FILE="$RUN_DIR/pid.txt"

OUTPUT_DIR="$RUN_DIR" nohup uv run python -m src.rl.train_grpo \
  --config "$CONFIG_YAML" \
  --output_dir "$RUN_DIR" \
  > "$LOG_FILE" 2>&1 &
PID=$!
printf "%s\n" "$PID" > "$PID_FILE"

sleep 1
if ! kill -0 "$PID" 2>/dev/null; then
  echo "failed to start" >&2
  exit 1
fi

echo "PID: $PID"
echo "Log: $LOG_FILE"
echo "Tail: tail -n 50 -f $LOG_FILE"
echo "Stop: kill $(cat $PID_FILE)"
echo "RUN_DIR: $RUN_DIR"
