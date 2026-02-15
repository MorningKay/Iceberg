#!/usr/bin/env bash
set -euo pipefail

# Usage: start_assistant_vllm_server.sh <MODEL_PATH> [HOST] [PORT] [TP]
MODEL_PATH="$1"
HOST="${2:-0.0.0.0}"
PORT="${3:-9101}"
TP="${4:-2}"

if [ -z "$MODEL_PATH" ]; then
  echo "Usage: $0 <MODEL_PATH> [HOST] [PORT] [TP]" >&2
  exit 1
fi

OUTPUT_BASE="outputs/assistant_servers"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="$OUTPUT_BASE/$TIMESTAMP"
mkdir -p "$RUN_DIR"

LOG_FILE="$RUN_DIR/vllm_${PORT}.log"
PID_FILE="$RUN_DIR/pid.txt"

nohup vllm serve "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "$TP" \
  > "$LOG_FILE" 2>&1 &

PID=$!
printf "%s\n" "$PID" > "$PID_FILE"

sleep 1
if ! kill -0 "$PID" 2>/dev/null; then
  echo "failed to start" >&2
  exit 1
fi

echo "Started assistant vLLM server."
echo "PID: $PID"
echo "Log: $LOG_FILE"
echo "Tail: tail -n 50 -f $LOG_FILE"
echo "Stop: kill $(cat $PID_FILE)"
echo "RUN_DIR: $RUN_DIR"
