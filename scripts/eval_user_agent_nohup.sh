#!/usr/bin/env bash
set -euo pipefail

# Users can set ASCEND_RT_VISIBLE_DEVICES to select NPU devices.

if [ $# -lt 3 ] || [ $# -gt 7 ]; then
  echo "Usage: $0 <BASE_MODEL> <ADAPTER_DIR> <DATA_JSON> [OUTPUT_BASE] [DEVICE] [MAX_SAMPLES] [MAX_STEPS]" >&2
  exit 1
fi

BASE_MODEL="$1"
ADAPTER_DIR="$2"
DATA_JSON="$3"
OUTPUT_BASE="${4:-outputs/user_agent_eval}"
DEVICE="${5:-auto}"
MAX_SAMPLES="${6:-}"
MAX_STEPS="${7:-}"

if [ "$DEVICE" != "auto" ] && [ "$DEVICE" != "cpu" ] && [ "$DEVICE" != "npu" ]; then
  echo "DEVICE must be one of: auto|cpu|npu" >&2
  exit 1
fi

if [ ! -f "$DATA_JSON" ]; then
  echo "DATA_JSON not found: $DATA_JSON" >&2
  exit 1
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="$OUTPUT_BASE/$TIMESTAMP"
mkdir -p "$RUN_DIR"

LOG_FILE="$RUN_DIR/eval.log"
PID_FILE="$RUN_DIR/pid.txt"
OUTPUT_JSON="$RUN_DIR/predictions.json"

ARGS=(
  --base_model "$BASE_MODEL"
  --adapter_dir "$ADAPTER_DIR"
  --data_json "$DATA_JSON"
  --output_json "$OUTPUT_JSON"
  --device "$DEVICE"
)

if [ -n "$MAX_SAMPLES" ]; then
  ARGS+=(--max_samples "$MAX_SAMPLES")
fi

if [ -n "$MAX_STEPS" ]; then
  ARGS+=(--max_steps_per_dialogue "$MAX_STEPS")
fi

nohup uv --project lf run python -m src.models.eval_user_agent "${ARGS[@]}" > "$LOG_FILE" 2>&1 &
PID=$!
printf "%s\n" "$PID" > "$PID_FILE"

sleep 1
if ! kill -0 "$PID" 2>/dev/null; then
  echo "Failed to start evaluation process" >&2
  exit 1
fi

echo "Started evaluation."
echo "PID: $PID"
echo "Log: $LOG_FILE"
echo "Tail: tail -n 50 -f $LOG_FILE"
echo "Stop: kill $(cat $PID_FILE)"
echo "Output: $OUTPUT_JSON"
