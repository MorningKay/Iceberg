# Iceberg

[English](README.md) | [中文](README_ZH.md)

> This version is adapted for **ASCEND-NPU**. Environment management uses `uv`; start by running `uv sync` to synchronize Python dependencies. In addition to uv-managed dependencies, install these target-runtime packages separately: `torch==2.7.1`, `torch_npu==2.7.1`, `vllm==0.11.0`, `vllm-ascend==0.11.0rc1`, and the project-specific TRL fork at <https://github.com/MorningKay/trl.git>.

Iceberg is a research project for Chinese multi-turn empathetic dialogue RL. It trains a 5-way Iceberg-layer classifier, optionally fine-tunes a user-agent simulator with LLaMA-Factory, and trains the main assistant model with TRL GRPO. The GRPO stage uses external vLLM services for online two-agent rollout and uses classifier-produced layer histories to compute rewards.

## Environment Setup

The root project uses `pyproject.toml` and `uv.lock`; Python must be `>=3.10,<3.11`.

```bash
uv sync
```

Install task-specific dependency groups as needed:

```bash
uv sync --group cls    # classifier training
uv sync --group rl     # TRL / GRPO training
uv sync --group dev    # development tools
```

The Ascend runtime packages are intentionally not locked in the main uv environment. Install these manually in the target NPU environment:

```bash
torch==2.7.1
torch_npu==2.7.1
vllm==0.11.0
vllm-ascend==0.11.0rc1
```

This project also depends on a modified TRL fork for the Iceberg GRPO tuple reward and rollout behavior. Install the fork in the same environment used for GRPO:

```bash
uv pip install git+https://github.com/MorningKay/trl.git
```

For vLLM, torch, and CANN compatibility, see:

https://docs.vllm.ai/projects/ascend/en/latest/community/versioning_policy.html

For vllm-ascend installation, see:

https://vllm-ascend-en-yikun.readthedocs.io/zh-cn/latest/installation.html

User-agent SFT uses a separate LLaMA-Factory uv project under `lf/`:

```bash
uv sync --project lf
```

## Project Layout

```text
configs/
  classifier/train.yaml      # BERT classifier training config
  rl/grpo.yaml               # GRPO, rollout, service ports, and reward config
  user_agent/sft.yaml        # LLaMA-Factory SFT template

scripts/
  run_classifier_nohup.sh            # start classifier training in background
  export_best_classifier.sh          # export best classifier checkpoint
  convert_user_agent_dataset.py      # source dialogue JSON -> ShareGPT SFT JSON
  run_user_agent_sft_nohup.sh        # start LLaMA-Factory SFT
  export_best_user_agent.sh          # export user-agent checkpoint
  eval_user_agent_nohup.sh           # optional user-agent evaluation
  convert_rl_dataset.py              # source dialogue JSON -> GRPO prompt JSON
  start_assistant_vllm_server.sh     # start assistant-side TRL vLLM server
  start_user_agent_vllm_server.sh    # start user-agent vLLM server + classifier sidecar
  train_grpo_accelerate.sh           # start distributed GRPO training with accelerate
  run_grpo_nohup.sh                  # single-process GRPO nohup launcher

src/
  data/iceberg_dataset.py            # classifier data loading and splitting
  trainer/train_classifier.py        # BERT classifier training entrypoint
  models/iceberg_classifier.py       # classifier inference wrapper
  models/infer_user_agent.py         # local user-agent LoRA inference helper
  models/eval_user_agent.py          # user-agent teacher-forcing evaluation
  serving/vllm_client.py             # OpenAI/vLLM and TRL vLLM clients
  serving/user_agent_server.py       # sidecar: user generation + classifier
  serving/user_agent_client.py       # training-side HTTP client for sidecar
  rl/grpo_rollout.py                 # multi-turn two-agent rollout
  rl/reward_fn.py                    # Iceberg reward and TRL tuple reward interface
  rl/train_grpo.py                   # TRL GRPO training entrypoint
```

`data/`, `models/`, and `outputs/` are runtime directories and are normally not committed. Scripts write logs, PID files, config snapshots, and checkpoints under `outputs/...`.

## Train the BERT Iceberg Classifier

The classifier entrypoint is `src.trainer.train_classifier`; the background wrapper is `scripts/run_classifier_nohup.sh`. The input dataset must be a top-level JSON list. `src/data/iceberg_dataset.py` extracts only `Speaker` turns with a valid `iceberg_layer`.

The fixed label set is:

```text
behavior -> 0
coping -> 1
feelings -> 2
feelings about feelings -> 3
perceptions -> 4
```

Start training:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
./scripts/run_classifier_nohup.sh data/data.json outputs/classifier configs/classifier/train.yaml
```

The run directory looks like `outputs/classifier/YYYYmmdd_HHMMSS/` and contains `train.log`, `pid.txt`, `config_snapshot.yaml`, `git_commit.txt`, `test_metrics.json`, `test_pred_vs_gold.json`, `label_mapping.json`, and checkpoints.

Export the best classifier to the default GRPO path:

```bash
./scripts/export_best_classifier.sh outputs/classifier/<RUN_DIR> models/iceberg_classifier
```

## Optional User-Agent Fine-Tuning

The user agent is a Speaker simulator used during GRPO rollout. This step is optional if you already have a user-agent model that vLLM can serve.

Convert the original dialogue dataset to ShareGPT format:

```bash
uv run python scripts/convert_user_agent_dataset.py \
  --input_json data/data.json \
  --output_json data/user_finetune.json
```

Start LLaMA-Factory SFT:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
./scripts/run_user_agent_sft_nohup.sh models/Qwen2.5-7B-Instruct data/user_finetune.json outputs/user_agent lora
```

The script renders a runtime config from `configs/user_agent/sft.yaml` and runs `uv --project lf run llamafactory-cli train ...`.

Export the selected user-agent checkpoint:

```bash
./scripts/export_best_user_agent.sh outputs/user_agent/<RUN_DIR> models/user_agent_best
```

Optional evaluation:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
./scripts/eval_user_agent_nohup.sh \
  models/Qwen2.5-7B-Instruct \
  models/user_agent_best \
  data/data.json
```

## Prepare GRPO Data

Convert source dialogue data into GRPO prompt JSON:

```bash
uv run python scripts/convert_rl_dataset.py \
  --input_json data/data.json \
  --output_json data/rl_prompts.json
```

The default GRPO config `configs/rl/grpo.yaml` uses:

```text
dataset_path: data/rl_prompts.json
model_name_or_path: models/Qwen2.5-7B-Instruct
classifier_checkpoint_dir: models/iceberg_classifier
user_agent_adapter_dir: models/user_agent_best
assistant_server_port: 9101
user_server_port: 9202
```

## Start GRPO Services and Training

GRPO uses three process groups:

1. Assistant-side TRL vLLM server for TRL server mode and weight sync.
2. User-agent vLLM server plus sidecar for user generation and `/classify`.
3. Main GRPO training process launched with accelerate.

Start the assistant server first. The script requires a model path; defaults are host `0.0.0.0`, port `9101`, and TP `2`.

```bash
ASCEND_RT_VISIBLE_DEVICES=4,5 \
./scripts/start_assistant_vllm_server.sh models/Qwen2.5-7B-Instruct 0.0.0.0 9101 2
```

Start the user-agent stack. This script starts OpenAI-compatible `vllm serve`, then starts `src.serving.user_agent_server`. The sidecar port is `PORT + 100`, so port `9102` maps to sidecar port `9202`.

```bash
ASCEND_RT_VISIBLE_DEVICES=6,7 \
./scripts/start_user_agent_vllm_server.sh <USER_AGENT_MODEL_PATH> models/iceberg_classifier 0.0.0.0 9102 2
```

After both services are up, start GRPO training:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
./scripts/train_grpo_accelerate.sh configs/rl/grpo.yaml 4
```

Training artifacts are written to `outputs/grpo/YYYYmmdd_HHMMSS/`. Stop training with:

```bash
kill -TERM -- -$(cat outputs/grpo/<RUN_DIR>/pgid.txt)
```

Service logs are under:

```text
outputs/assistant_servers/<RUN_DIR>/
outputs/user_servers/<RUN_DIR>/
```

Use the `Stop:` command printed by each startup script to stop services.

## GRPO Runtime Notes

`src/rl/train_grpo.py` wires:

```text
rollout_func -> src/rl/grpo_rollout.py
reward_funcs -> src/rl/reward_fn.py
```

Rollout keeps two histories: `H_main` for the trainable assistant policy, and `H_user` for the user-agent simulator. The classifier writes `layer_history` entries, and `src/rl/reward_fn.py` returns `(rewards, extra)`. `extra["reward_term_means"]` is consumed by the local TRL logging path and appears as Trainer/W&B metrics.

## End-to-End Workflow

```bash
# 1. Sync project environments
uv sync
uv sync --group cls
uv sync --group rl

# 2. Install Ascend runtime packages manually
# torch==2.7.1, torch_npu==2.7.1, vllm==0.11.0, vllm-ascend==0.11.0rc1

# 3. Train and export classifier
ASCEND_RT_VISIBLE_DEVICES=0 ./scripts/run_classifier_nohup.sh data/data.json
./scripts/export_best_classifier.sh outputs/classifier/<RUN_DIR> models/iceberg_classifier

# 4. Optional user-agent SFT
uv run python scripts/convert_user_agent_dataset.py --input_json data/data.json --output_json data/user_finetune.json
ASCEND_RT_VISIBLE_DEVICES=0 ./scripts/run_user_agent_sft_nohup.sh models/Qwen2.5-7B-Instruct data/user_finetune.json
./scripts/export_best_user_agent.sh outputs/user_agent/<RUN_DIR> models/user_agent_best

# 5. Prepare RL prompts
uv run python scripts/convert_rl_dataset.py --input_json data/data.json --output_json data/rl_prompts.json

# 6. Start services and train
ASCEND_RT_VISIBLE_DEVICES=4,5 ./scripts/start_assistant_vllm_server.sh models/Qwen2.5-7B-Instruct 0.0.0.0 9101 2
ASCEND_RT_VISIBLE_DEVICES=6,7 ./scripts/start_user_agent_vllm_server.sh <USER_AGENT_MODEL_PATH> models/iceberg_classifier 0.0.0.0 9102 2
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 ./scripts/train_grpo_accelerate.sh configs/rl/grpo.yaml 4
```

`scripts/start_user_agent_vllm_server.sh` expects `<USER_AGENT_MODEL_PATH>` to be directly loadable by vLLM. If you only have a LoRA adapter, prepare a serveable model form before passing it to this script.
