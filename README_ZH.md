# Iceberg

[English](README.md) | [中文](README_ZH.md)

> 本版本适配 **ASCEND-NPU**。环境管理使用 `uv`，首次使用请先运行 `uv sync` 同步 Python 依赖。除 `uv` 管理的依赖外，还需要在目标 Ascend 环境中单独安装：`torch==2.7.1`、`torch_npu==2.7.1`、`vllm==0.11.0`、`vllm-ascend==0.11.0rc1`，以及项目使用的自定义 TRL fork：<https://github.com/MorningKay/trl.git>。

Iceberg 是一个面向中文多轮共情对话的研究项目。项目包含三部分：训练 5 类冰山层级 BERT 分类器、可选地用 LLaMA-Factory 微调用户代理、最后用 TRL GRPO 训练主助手模型。GRPO 阶段使用外部 vLLM/TRL vLLM 服务做在线多轮 rollout，并用分类器输出的 iceberg layer 序列计算奖励。

## 环境设置

项目主环境由根目录的 `pyproject.toml` 和 `uv.lock` 管理，Python 版本要求为 `>=3.10,<3.11`。

```bash
uv sync
```

按任务同步依赖组：

```bash
uv sync --group cls    # 分类器训练
uv sync --group rl     # TRL / GRPO 训练
uv sync --group dev    # 开发工具
```

本仓库不把 Ascend 运行时的大包写进主依赖锁中。请在目标 NPU 环境中按平台要求单独安装：

```bash
torch==2.7.1
torch_npu==2.7.1
vllm==0.11.0
vllm-ascend==0.11.0rc1
```

本项目还依赖修改过的 TRL fork，用于 Iceberg GRPO 的 tuple reward 和 rollout 行为。请在 GRPO 使用的同一个环境中安装：

```bash
uv pip install git+https://github.com/MorningKay/trl.git
```

vLLM、torch、CANN 的版本兼容关系见：

https://docs.vllm.ai/projects/ascend/en/latest/community/versioning_policy.html

vllm-ascend 安装流程见：

https://vllm-ascend-en-yikun.readthedocs.io/zh-cn/latest/installation.html

用户代理 SFT 使用独立的 LLaMA-Factory `uv` 项目，位于 `lf/`：

```bash
uv sync --project lf
```

## 项目结构

```text
configs/
  classifier/train.yaml      # BERT 分类器训练配置
  rl/grpo.yaml               # GRPO、rollout、远程服务端口和 reward 配置
  user_agent/sft.yaml        # LLaMA-Factory SFT 配置模板

scripts/
  run_classifier_nohup.sh            # 后台启动分类器训练
  export_best_classifier.sh          # 导出分类器最佳 checkpoint
  convert_user_agent_dataset.py      # 原始对话数据 -> ShareGPT 用户代理 SFT 数据
  run_user_agent_sft_nohup.sh        # 后台启动 LLaMA-Factory SFT
  export_best_user_agent.sh          # 导出用户代理 checkpoint
  eval_user_agent_nohup.sh           # 可选用户代理评估
  convert_rl_dataset.py              # 原始对话数据 -> GRPO prompt JSON
  start_assistant_vllm_server.sh     # 启动助手侧 TRL vLLM server
  start_user_agent_vllm_server.sh    # 启动用户代理 vLLM server + classifier sidecar
  train_grpo_accelerate.sh           # accelerate 多进程启动 GRPO 训练
  run_grpo_nohup.sh                  # 单进程 nohup GRPO 启动脚本

src/
  data/iceberg_dataset.py            # 分类器数据读取和切分
  trainer/train_classifier.py        # BERT 分类器训练入口
  models/iceberg_classifier.py       # 分类器推理封装
  models/infer_user_agent.py         # 本地用户代理 LoRA 推理工具
  models/eval_user_agent.py          # 用户代理 teacher-forcing 评估
  serving/vllm_client.py             # OpenAI/vLLM 和 TRL vLLM 客户端
  serving/user_agent_server.py       # 用户代理 sidecar：生成 + 分类
  serving/user_agent_client.py       # 训练侧调用用户代理 sidecar 的 HTTP 客户端
  rl/grpo_rollout.py                 # 多轮两代理 rollout
  rl/reward_fn.py                    # Iceberg reward 计算和 TRL tuple reward 接口
  rl/train_grpo.py                   # TRL GRPO 训练入口
```

`data/`、`models/`、`outputs/` 是运行时目录，通常不提交到 Git。当前脚本默认把训练日志、PID、配置快照和 checkpoint 写入 `outputs/...`。

## 训练 BERT 冰山分类器

分类器训练入口是 `src.trainer.train_classifier`，后台脚本是 `scripts/run_classifier_nohup.sh`。输入数据必须是顶层 JSON list；加载逻辑在 `src/data/iceberg_dataset.py` 中，只抽取 `dialogue[].turns[]` 里 `role == "Speaker"` 且带有合法 `iceberg_layer` 的文本。

标签固定为 5 类：

```text
behavior -> 0
coping -> 1
feelings -> 2
feelings about feelings -> 3
perceptions -> 4
```

启动训练：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
./scripts/run_classifier_nohup.sh data/data.json outputs/classifier configs/classifier/train.yaml
```

训练输出目录形如 `outputs/classifier/YYYYmmdd_HHMMSS/`，包含 `train.log`、`pid.txt`、`config_snapshot.yaml`、`git_commit.txt`、`test_metrics.json`、`test_pred_vs_gold.json`、`label_mapping.json` 和 checkpoint。

导出最佳分类器到 GRPO 默认读取位置：

```bash
./scripts/export_best_classifier.sh outputs/classifier/<RUN_DIR> models/iceberg_classifier
```

## 可选：微调用户代理

用户代理是 Speaker simulator，用于 GRPO rollout 中生成下一轮用户发言。此步骤是可选的；如果已有可被 vLLM 加载的用户代理模型，可以直接进入服务启动。

先把原始对话数据转换为 ShareGPT JSON：

```bash
uv run python scripts/convert_user_agent_dataset.py \
  --input_json data/data.json \
  --output_json data/user_finetune.json
```

启动 LLaMA-Factory SFT：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
./scripts/run_user_agent_sft_nohup.sh models/Qwen2.5-7B-Instruct data/user_finetune.json outputs/user_agent lora
```

该脚本会基于 `configs/user_agent/sft.yaml` 生成运行时配置，并通过 `uv --project lf run llamafactory-cli train ...` 启动训练。

导出用户代理 checkpoint：

```bash
./scripts/export_best_user_agent.sh outputs/user_agent/<RUN_DIR> models/user_agent_best
```

可选评估：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
./scripts/eval_user_agent_nohup.sh \
  models/Qwen2.5-7B-Instruct \
  models/user_agent_best \
  data/data.json
```

## 准备 GRPO 数据

GRPO 数据转换脚本是 `scripts/convert_rl_dataset.py`。它从原始对话数据中取第一轮 Speaker 文本作为主模型初始 user prompt，并保留 `first_explanation` 和 `dia_id`。

```bash
uv run python scripts/convert_rl_dataset.py \
  --input_json data/data.json \
  --output_json data/rl_prompts.json
```

默认 GRPO 配置 `configs/rl/grpo.yaml` 读取：

```text
dataset_path: data/rl_prompts.json
model_name_or_path: models/Qwen2.5-7B-Instruct
classifier_checkpoint_dir: models/iceberg_classifier
user_agent_adapter_dir: models/user_agent_best
assistant_server_port: 9101
user_server_port: 9202
```

## 启动多服务 GRPO 训练

GRPO 运行依赖三个进程组：

1. 助手侧 TRL vLLM server：服务 TRL GRPO 的 server mode 和权重同步。
2. 用户代理 vLLM server + sidecar：生成 user utterance，并通过同一 sidecar 调用分类器 `/classify`。
3. 主 GRPO 训练进程：使用 accelerate 在训练 NPU 上做 LoRA GRPO。

先启动助手侧服务。脚本要求传入模型路径，默认端口是 `9101`，默认 TP 是 `2`：

```bash
ASCEND_RT_VISIBLE_DEVICES=4,5 \
./scripts/start_assistant_vllm_server.sh models/Qwen2.5-7B-Instruct 0.0.0.0 9101 2
```

再启动用户代理服务。脚本会先启动 OpenAI-compatible `vllm serve`，再启动 `src.serving.user_agent_server` sidecar；sidecar 端口是 vLLM 端口 `+100`。例如 vLLM 端口 `9102` 时，训练配置中的 `user_server_port` 应为 `9202`。

```bash
ASCEND_RT_VISIBLE_DEVICES=6,7 \
./scripts/start_user_agent_vllm_server.sh <USER_AGENT_MODEL_PATH> models/iceberg_classifier 0.0.0.0 9102 2
```

确认两个服务启动后，再启动 GRPO 训练。默认配置已使用 `vllm_mode: server`，并连接 `127.0.0.1:9101` 和 `127.0.0.1:9202`：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
./scripts/train_grpo_accelerate.sh configs/rl/grpo.yaml 4
```

训练日志和 PID 会写到 `outputs/grpo/YYYYmmdd_HHMMSS/`。停止训练：

```bash
kill -TERM -- -$(cat outputs/grpo/<RUN_DIR>/pgid.txt)
```

停止服务时使用启动脚本输出的 `Stop:` 命令。服务日志位于：

```text
outputs/assistant_servers/<RUN_DIR>/
outputs/user_servers/<RUN_DIR>/
```

## GRPO 训练逻辑

`src/rl/train_grpo.py` 接入：

```text
rollout_func -> src/rl/grpo_rollout.py
reward_funcs -> src/rl/reward_fn.py
```

rollout 阶段维护两个历史：

```text
H_main: 给主助手模型看的 system/user/assistant 对话历史
H_user: 给用户代理看的对话历史，主助手输出映射为 user，用户代理输出映射为 assistant
```

分类器输出被写入 `layer_history`，reward 函数返回 `(rewards, extra)`。`extra["reward_term_means"]` 中的 reward term 会由当前 TRL logging path 记录为 Trainer 指标和 W&B 指标。

## 端到端流程摘要

```bash
# 1. 同步主环境
uv sync
uv sync --group cls
uv sync --group rl

# 2. 单独安装 Ascend 运行时依赖
# torch==2.7.1, torch_npu==2.7.1, vllm==0.11.0, vllm-ascend==0.11.0rc1

# 3. 训练并导出分类器
ASCEND_RT_VISIBLE_DEVICES=0 ./scripts/run_classifier_nohup.sh data/data.json
./scripts/export_best_classifier.sh outputs/classifier/<RUN_DIR> models/iceberg_classifier

# 4. 可选：转换数据并微调用户代理
uv run python scripts/convert_user_agent_dataset.py --input_json data/data.json --output_json data/user_finetune.json
ASCEND_RT_VISIBLE_DEVICES=0 ./scripts/run_user_agent_sft_nohup.sh models/Qwen2.5-7B-Instruct data/user_finetune.json
./scripts/export_best_user_agent.sh outputs/user_agent/<RUN_DIR> models/user_agent_best

# 5. 转换 GRPO 数据
uv run python scripts/convert_rl_dataset.py --input_json data/data.json --output_json data/rl_prompts.json

# 6. 启动服务和训练
ASCEND_RT_VISIBLE_DEVICES=4,5 ./scripts/start_assistant_vllm_server.sh models/Qwen2.5-7B-Instruct 0.0.0.0 9101 2
ASCEND_RT_VISIBLE_DEVICES=6,7 ./scripts/start_user_agent_vllm_server.sh <USER_AGENT_MODEL_PATH> models/iceberg_classifier 0.0.0.0 9102 2
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 ./scripts/train_grpo_accelerate.sh configs/rl/grpo.yaml 4
```

注意：`scripts/start_user_agent_vllm_server.sh` 的第一个参数必须是 vLLM 能直接加载的用户代理模型路径。若只导出了 LoRA adapter，需要先准备可服务化的模型形态，再传给该脚本。
