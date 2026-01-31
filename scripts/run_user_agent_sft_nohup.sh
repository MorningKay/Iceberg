#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ] || [ $# -gt 4 ]; then
  echo "Usage: $0 <BASE_MODEL> <DATA_JSON> [OUTPUT_BASE] [MODE]" >&2
  exit 1
fi

BASE_MODEL="$1"
DATA_JSON="$2"
OUTPUT_BASE="${3:-outputs/user_agent}"
MODE="${4:-lora}"

if [ "$MODE" != "lora" ] && [ "$MODE" != "full" ]; then
  echo "MODE must be 'lora' or 'full'" >&2
  exit 1
fi

if [ ! -f "$DATA_JSON" ]; then
  echo "DATA_JSON not found: $DATA_JSON" >&2
  exit 1
fi

TEMPLATE_CONFIG="configs/user_agent/sft.yaml"
if [ ! -f "$TEMPLATE_CONFIG" ]; then
  echo "Template config not found: $TEMPLATE_CONFIG" >&2
  exit 1
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="$OUTPUT_BASE/$TIMESTAMP"
mkdir -p "$RUN_DIR"

LOG_FILE="$RUN_DIR/train.log"
PID_FILE="$RUN_DIR/pid.txt"
RUNTIME_CONFIG="$RUN_DIR/sft_runtime.yaml"

DATASET_DIR=$(dirname "$DATA_JSON")
DATASET_NAME=$(basename "$DATA_JSON")

MAX_SAMPLES_VALUE="${MAX_SAMPLES:-}"

if [ -n "$MAX_SAMPLES_VALUE" ]; then
  MAX_SAMPLES_LINE="max_samples: $MAX_SAMPLES_VALUE"
else
  MAX_SAMPLES_LINE="max_samples: 500000"
fi

sed \
  -e "s|MODEL_NAME_OR_PATH_PLACEHOLDER|$BASE_MODEL|g" \
  -e "s|OUTPUT_DIR_PLACEHOLDER|$RUN_DIR|g" \
  -e "s|DATASET_DIR_PLACEHOLDER|$DATASET_DIR|g" \
  -e "s|^max_samples: .*|$MAX_SAMPLES_LINE|" \
  "$TEMPLATE_CONFIG" > "$RUNTIME_CONFIG"

if [ "$MODE" = "full" ]; then
  sed -i \
    -e "s|^finetuning_type:.*|finetuning_type: full|" \
    -e "/^lora_/d" \
    -e "/^lora_target:/d" \
    "$RUNTIME_CONFIG"
fi

DATASET_INFO="$DATASET_DIR/dataset_info.json"
cat > "$DATASET_INFO" <<EOF
{
  "user_agent_sharegpt": {
    "file_name": "$DATASET_NAME",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "role": "from",
      "content": "value"
    }
  }
}
EOF

nohup uv --directory lf run llamafactory-cli train "$RUNTIME_CONFIG" > "$LOG_FILE" 2>&1 &

PID=$!
printf "%s\n" "$PID" > "$PID_FILE"

echo "Started SFT."
echo "PID: $PID"
echo "Log: $LOG_FILE"
echo "Tail: tail -n 50 -f $LOG_FILE"
echo "Stop: kill $PID"
