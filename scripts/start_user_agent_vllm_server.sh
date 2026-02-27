#!/usr/bin/env bash
set -euo pipefail

# Usage: start_user_agent_vllm_server.sh <MODEL_PATH> <CLASSIFIER_DIR> [HOST] [PORT] [TP]
MODEL_PATH="${1:-}"
CLASSIFIER_DIR="${2:-}"
HOST="${3:-0.0.0.0}"
PORT="${4:-9102}"
TP="${5:-2}"

if [ -z "$MODEL_PATH" ] || [ -z "$CLASSIFIER_DIR" ]; then
  echo "Usage: $0 <MODEL_PATH> <CLASSIFIER_DIR> [HOST] [PORT] [TP]" >&2
  exit 1
fi

OUTPUT_BASE="outputs/user_servers"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="$OUTPUT_BASE/$TIMESTAMP"
mkdir -p "$RUN_DIR"

VLLM_LOG="$RUN_DIR/user_vllm_${PORT}.log"
SVC_LOG="$RUN_DIR/svc_vllm_${PORT}.log"
VLLM_PGID="$RUN_DIR/user_pgid.txt"
SVC_PGID="$RUN_DIR/svc_pgid.txt"

# NOTE:
# - Assistant-side server MUST use `trl vllm-serve` (TRL weight sync / communicator).
# - User-agent side uses OpenAI-compatible `vllm serve` (simple chat/completions API).
setsid vllm serve "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "$TP" \
  > "$VLLM_LOG" 2>&1 &

VLLM_PID_VAL=$!
VLLM_PGID_VAL="$(ps -o pgid= -p "$VLLM_PID_VAL" | tr -d ' ')"
printf "%s\n" "$VLLM_PGID_VAL" > "$VLLM_PGID"

sleep 1
if ! kill -0 "$VLLM_PID_VAL" 2>/dev/null; then
  echo "vLLM failed to start" >&2
  exit 1
fi

# Sidecar classifier + user agent proxy on same host, different port.
CLIENT_HOST="${CLIENT_HOST:-127.0.0.1}"
SVC_PORT=$((PORT + 100))
setsid uv run python -m src.serving.user_agent_server \
  --host "$CLIENT_HOST" \
  --port "$SVC_PORT" \
  --vllm_url "http://$CLIENT_HOST:$PORT" \
  --vllm_model "$MODEL_PATH" \
  --classifier_dir "$CLASSIFIER_DIR" \
  > "$SVC_LOG" 2>&1 &

SVC_PID_VAL=$!
SVC_PGID_VAL="$(ps -o pgid= -p "$SVC_PID_VAL" | tr -d ' ')"
printf "%s\n" "$SVC_PGID_VAL" > "$SVC_PGID"

sleep 1
if ! kill -0 "$SVC_PID_VAL" 2>/dev/null; then
  echo "user-agent service failed to start" >&2
  exit 1
fi

echo "Started user-agent vLLM server."
echo "vLLM Log: $VLLM_LOG"
echo "vLLM Tail: tail -n 50 -f $VLLM_LOG"
echo "vLLM Stop: kill -TERM -- -$VLLM_PGID_VAL"

echo "User-agent service"
echo "Service Log: $SVC_LOG"
echo "Service tail: tail -n 50 -f $SVC_LOG"
echo "Service Stop: kill -TERM -- -$SVC_PGID_VAL"

echo "[user] vLLM bind:  http://$HOST:$PORT"
echo "[user] vLLM url:   http://$CLIENT_HOST:$PORT"
echo "[user] sidecar:    http://$CLIENT_HOST:$((PORT+100))"

echo "RUN_DIR: $RUN_DIR"
