#!/usr/bin/env bash
set -euo pipefail

# Usage: train_grpo_accelerate.sh <CONFIG_YAML> [NUM_PROCESSES]
CONFIG_YAML="$1"
NUM_PROCESSES="${2:-4}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="$OUTPUT_BASE/$TIMESTAMP"
mkdir -p "$RUN_DIR"

cp "$CONFIG_YAML" "$RUN_DIR/config_snapshot.yaml"

LOG_FILE="$RUN_DIR/train.log"
PID_FILE="$RUN_DIR/pid.txt"

if [ -z "$CONFIG_YAML" ]; then
  echo "Usage: $0 <CONFIG_YAML> [NUM_PROCESSES]" >&2
  exit 1
fi

nohup uv run accelerate launch --num_processes "$NUM_PROCESSES" \
  -m src.rl.train_grpo --config "$CONFIG_YAML" \
  > "$LOG_FILE" 2>&1 &

PID=$!
printf "%s\n" "$PID" > "$PID_FILE"

sleep 1
if ! kill -0 "$PID" 2>/dev/null; then
  echo "failed to start" >&2
  exit 1
fi

echo "Training started."
echo "Log: $LOG_FILE"
echo "Tail: tail -n 50 -f $LOG_FILE"
echo "Stop: kill $(cat $PID_FILE)"
echo "RUN_DIR: $RUN_DIR"
