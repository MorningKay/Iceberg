#!/usr/bin/env bash
set -euo pipefail

DATA_PATH="$1"
OUTPUT_BASE="${2:-outputs/classifier}"
CONFIG_PATH="${3:-configs/classifier/train.yaml}"

if [ -z "$DATA_PATH" ]; then
  echo "Usage: $0 <data_json> [output_base_dir] [config_yaml]" >&2
  exit 1
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="$OUTPUT_BASE/$TIMESTAMP"
mkdir -p "$RUN_DIR"

LOG_FILE="$RUN_DIR/train.log"

nohup uv run python -m src.trainer.train_classifier \
  --data "$DATA_PATH" \
  --output_dir "$RUN_DIR" \
  --config "$CONFIG_PATH" \
  > "$LOG_FILE" 2>&1 &

PID=$!
echo "$PID" > "$RUN_DIR/pid.txt"
echo "Started training. PID: $PID"
echo "Logs: tail -f $LOG_FILE"
echo "Stop: kill \$(cat $RUN_DIR/pid.txt)"
