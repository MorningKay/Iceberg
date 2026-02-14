"""GRPO training entrypoint for multi-turn iceberg dialogue RL."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import os
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from src.models.infer_user_agent import load_user_agent
from src.rl.grpo_rollout import RolloutConfig, run_episode
from src.models.iceberg_classifier import IcebergClassifier
from src.rl.reward_fn import RewardConfig, trl_reward_func


@dataclass
class TrainConfig:
    model_name_or_path: str
    user_agent_base_model: str
    user_agent_adapter_dir: str
    classifier_checkpoint_dir: str
    dataset_path: str
    reward: Dict[str, Any] = None
    max_turns: int = 8
    user_max_new_tokens: int = 128
    user_temperature: float = 0.7
    user_top_p: float = 0.9
    user_do_sample: bool = True
    main_max_new_tokens: int = 128
    main_temperature: float = 0.7
    main_top_p: float = 0.9
    main_do_sample: bool = True
    num_generations: int = 4
    use_vllm: bool = True
    vllm_mode: str = "colocate"
    vllm_gpu_memory_utilization: float | None = None
    vllm_max_model_length: int | None = None
    vllm_tensor_parallel_size: int | None = None
    vllm_host: str | None = None
    vllm_port: int | None = None
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-6
    max_steps: int = 1000
    logging_steps: int = 10
    save_steps: int = 100
    seed: int = 42
    report_to: List[str] = None
    run_name: str = "grpo"
    disable_tqdm: bool = False
    log_level: str = "info"
    wandb_project: str = "Iceberg-GRPO"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = None


def _load_yaml(path: str) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _build_train_config(config_path: str) -> TrainConfig:
    data = _load_yaml(config_path)
    cfg = TrainConfig(
        model_name_or_path=data["model_name_or_path"],
        user_agent_base_model=data["user_agent_base_model"],
        user_agent_adapter_dir=data["user_agent_adapter_dir"],
        classifier_checkpoint_dir=data["classifier_checkpoint_dir"],
        dataset_path=data["dataset_path"],
    )
    for key, value in data.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    if cfg.lora_target_modules is None:
        cfg.lora_target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    if cfg.report_to is None:
        cfg.report_to = ["wandb"]
    if cfg.reward is None:
        cfg.reward = {}
    return cfg


def _rollout_func(prompts, trainer, **kwargs):
    main_model = trainer.model
    main_tokenizer = trainer.processing_class
    user_model, user_tokenizer, _ = trainer.user_agent_bundle
    classifier = trainer.classifier
    rollout_cfg = trainer.rollout_config
    num_generations = int(getattr(trainer.args, "num_generations", 1))

    results = {
        "prompt_ids": [],
        "completion_ids": [],
        "logprobs": [],
        "layer_history": [],
        "terminate_reason": [],
        "num_user_turns": [],
        "reward_config": [],
        "reward_defaults": [],
        "dia_id": [],
    }

    for idx, prompt in enumerate(prompts):
        prompt_messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": str(prompt)}]
        first_explanations = kwargs.get("first_explanation")
        first_explanation = first_explanations[idx] if first_explanations is not None else ""
        dia_ids = kwargs.get("dia_id")
        dia_id = dia_ids[idx] if dia_ids is not None else None

        for _ in range(num_generations):
            episode = run_episode(
                prompt_messages=prompt_messages,
                first_explanation=first_explanation,
                main_model=main_model,
                main_tokenizer=main_tokenizer,
                user_model=user_model,
                user_tokenizer=user_tokenizer,
                classifier=classifier,
                cfg=rollout_cfg,
            )
            results["prompt_ids"].append(episode.get("prompt_ids", []))
            results["completion_ids"].append(episode.get("completion_ids", []))
            results["logprobs"].append(episode.get("logprobs", []))
            results["layer_history"].append(episode.get("layer_history", []))
            results["terminate_reason"].append(episode.get("terminate_reason", ""))
            results["num_user_turns"].append(episode.get("num_user_turns", 0))
            results["reward_config"].append({})
            results["reward_defaults"].append(trainer.reward_defaults.__dict__)
            results["dia_id"].append(dia_id)

    expected = len(prompts) * num_generations
    assert len(results["completion_ids"]) == expected
    assert len(results["layer_history"]) == expected
    assert len(results["num_user_turns"]) == expected
    if expected > 0:
        print(
            "rollout_func lengths:",
            {k: len(v) for k, v in results.items()},
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rl/grpo.yaml")
    parser.add_argument("--output_dir", help="Output directory")
    args = parser.parse_args()

    cfg = _build_train_config(args.config)

    output_dir = args.output_dir

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path)

    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg.lora_target_modules,
    )

    dataset = load_dataset("json", data_files=cfg.dataset_path, split="train")

    grpo_args = GRPOConfig(
        output_dir=output_dir,
        num_generations=cfg.num_generations,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        max_steps=cfg.max_steps,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        seed=cfg.seed,
        report_to=cfg.report_to,
        run_name=cfg.run_name,
        disable_tqdm=cfg.disable_tqdm,
        log_level=cfg.log_level,
        remove_unused_columns=False,
        use_vllm=cfg.use_vllm,
        vllm_mode=cfg.vllm_mode,
        vllm_gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
        vllm_max_model_length=cfg.vllm_max_model_length,
        vllm_tensor_parallel_size=cfg.vllm_tensor_parallel_size,
        vllm_host=cfg.vllm_host,
        vllm_port=cfg.vllm_port,
    )

    base_reward_cfg = RewardConfig()
    for key, value in cfg.reward.items():
        if hasattr(base_reward_cfg, key):
            setattr(base_reward_cfg, key, value)

    trainer = GRPOTrainer(
        model=model,
        args=grpo_args,
        train_dataset=dataset,
        reward_funcs=trl_reward_func,
        rollout_func=_rollout_func,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    if cfg.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", cfg.wandb_project)

    trainer.user_agent_bundle = load_user_agent(
        cfg.user_agent_base_model,
        cfg.user_agent_adapter_dir,
        device="auto",
        local_files_only=True,
    )
    trainer.classifier = IcebergClassifier(
        cfg.classifier_checkpoint_dir,
        device="auto",
        local_files_only=True,
    )
    trainer.rollout_config = RolloutConfig(
        max_turns=cfg.max_turns,
        user_max_new_tokens=cfg.user_max_new_tokens,
        user_temperature=cfg.user_temperature,
        user_top_p=cfg.user_top_p,
        user_do_sample=cfg.user_do_sample,
        main_max_new_tokens=cfg.main_max_new_tokens,
        main_temperature=cfg.main_temperature,
        main_top_p=cfg.main_top_p,
        main_do_sample=cfg.main_do_sample,
    )
    trainer.reward_defaults = base_reward_cfg

    trainer.train()


if __name__ == "__main__":
    main()
