#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
  echo "Usage: $0 <RUN_DIR> [DEST_DIR]" >&2
  exit 1
fi

RUN_DIR="$1"
DEST_DIR="${2:-models/user_agent_best}"

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
    uv run python - "$state_file" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p, "r", encoding="utf-8"))
best = None
for e in d.get("log_history", []):
    v = e.get("eval_loss")
    if isinstance(v, (int, float)) and (best is None or v < best):
        best = v

ckpt_dir = p.rsplit("/", 1)[0]
if best is None:
    print("unknown\t" + ckpt_dir)
else:
    print(f"{best}\t{ckpt_dir}")
PY
  done | awk -F '\t' 'NF==2 && $1!="unknown" {print $0}' | sort -t $'\t' -k1,1g | head -n 1
)

BEST_EVAL_LOSS=""
BEST_CHECKPOINT=""

if [ -n "$BEST_LINE" ]; then
  BEST_EVAL_LOSS=$(printf "%s" "$BEST_LINE" | awk -F '\t' '{print $1}')
  BEST_CHECKPOINT=$(printf "%s" "$BEST_LINE" | awk -F '\t' '{print $2}')
  CRITERION="eval_loss"
else
  CRITERION="eval_loss"
  BEST_EVAL_LOSS="unknown"
  BEST_CHECKPOINT=$(
    for state_file in "${STATE_FILES[@]}"; do
      checkpoint_dir=$(dirname "$state_file")
      step=${checkpoint_dir##*/checkpoint-}
      printf "%s\t%s\n" "$step" "$checkpoint_dir"
    done | sort -t $'\t' -k1,1nr | head -n 1 | awk -F '\t' '{print $2}'
  )
fi

if [ -z "$BEST_CHECKPOINT" ] || [ ! -d "$BEST_CHECKPOINT" ]; then
  echo "Failed to select a checkpoint under $RUN_DIR" >&2
  exit 1
fi

if [ -e "$DEST_DIR" ]; then
  rm -rf "$DEST_DIR"
fi

mkdir -p "$(dirname "$DEST_DIR")"
cp -a "$BEST_CHECKPOINT" "$DEST_DIR"

EXPORT_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
cat > "$DEST_DIR/source_info.txt" <<EOF
run_dir=$RUN_DIR
best_checkpoint=$BEST_CHECKPOINT
criterion=$CRITERION
best_eval_loss=$BEST_EVAL_LOSS
exported_at=$EXPORT_TIME
EOF

echo "selected_checkpoint=$BEST_CHECKPOINT"
echo "best_eval_loss=$BEST_EVAL_LOSS"
echo "dest_dir=$DEST_DIR"
