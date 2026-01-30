#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
  echo "Usage: $0 <RUN_DIR> [DEST_DIR]" >&2
  exit 1
fi

RUN_DIR="$1"
DEST_DIR="${2:-models/iceberg_classifier}"

if [ ! -d "$RUN_DIR" ]; then
  echo "RUN_DIR not found: $RUN_DIR" >&2
  exit 1
fi

shopt -s nullglob
STATE_FILES=("$RUN_DIR"/checkpoint-*/trainer_state.json)
shopt -u nullglob

if [ ${#STATE_FILES[@]} -eq 0 ]; then
  echo "No trainer_state.json found under $RUN_DIR/checkpoint-*" >&2
  exit 1
fi

BEST_LINE=$(
  for state_file in "${STATE_FILES[@]}"; do
    metric=$(uv run python -c "import json,sys; data=json.load(open(sys.argv[1])); print(data['best_metric'])" "$state_file")
    checkpoint_dir=$(dirname "$state_file")
    printf "%s\t%s\n" "$metric" "$checkpoint_dir"
  done | awk 'NR==1{best=$1; path=$2} NR>1{if($1>best){best=$1; path=$2}} END{if(NR>0){print best"\t"path}}'
)

if [ -z "$BEST_LINE" ]; then
  echo "Failed to select best checkpoint under $RUN_DIR" >&2
  exit 1
fi

BEST_METRIC=$(printf "%s" "$BEST_LINE" | awk -F '\t' '{print $1}')
BEST_CHECKPOINT=$(printf "%s" "$BEST_LINE" | awk -F '\t' '{print $2}')

if [ ! -d "$BEST_CHECKPOINT" ]; then
  echo "Best checkpoint directory not found: $BEST_CHECKPOINT" >&2
  exit 1
fi

if [ -e "$DEST_DIR" ]; then
  rm -rf "$DEST_DIR"
fi

mkdir -p "$(dirname "$DEST_DIR")"
cp -a "$BEST_CHECKPOINT" "$DEST_DIR"

{
  echo "run_dir=$RUN_DIR"
  echo "best_checkpoint=$BEST_CHECKPOINT"
  echo "best_metric=$BEST_METRIC"
  echo "exported_at=$(date -Is)"
} > "$DEST_DIR/source_info.txt"

echo "selected_checkpoint=$BEST_CHECKPOINT"
echo "best_metric=$BEST_METRIC"
echo "dest_dir=$DEST_DIR"
