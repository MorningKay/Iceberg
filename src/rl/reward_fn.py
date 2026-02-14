"""Reward computation for multi-turn iceberg dialogue RL."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RewardConfig:
    beta: float = 0.2
    delta: float = 0.5
    kappa: float = 10.0
    c: float = 0.5
    stage_penalty: float = 0.4
    delta_rep: float = 0.1
    rep_window: int = 5
    k2: float = 4.0
    c2: float = 0.0
    max_depth: int = 5
    bonus_depth: float = 0.0
    bonus_progress: float = 0.0
    bonus_progress_min: int = 0
    penalty_empty_user: float = 0.0
    penalty_early_terminate: float = 0.0


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _expected_layer(k: int, k_total: int, max_depth: int) -> int:
    step = math.ceil(k_total / max_depth)
    return min(max_depth, 1 + (k - 1) // step)


def compute_episode_reward(
    layer_history: List[Dict[str, Any]],
    num_user_turns: int,
    cfg: RewardConfig,
    terminate_reason: Optional[str] = None,
) -> float:
    if num_user_turns <= 0:
        return 0.0

    rewards: List[float] = []
    progress_count = 0
    deepest_layer = 0

    for idx, item in enumerate(layer_history, start=1):
        layer = int(item["layer"])
        confidence = float(item["confidence"])
        prev_layer = int(layer_history[idx - 2]["layer"]) if idx > 1 else layer

        deepest_layer = max(deepest_layer, layer)

        u = idx / float(num_user_turns)
        stage = _sigmoid(cfg.kappa * (u - cfg.c))

        base_shallow = min(layer, 2.0)
        base_deep = max(layer - 2.0, 0.0)
        r_base = confidence * ((1 - stage) * base_shallow + stage * base_deep) / cfg.max_depth

        r_prog = cfg.beta if layer == prev_layer + 1 else 0.0
        if r_prog > 0:
            progress_count += 1

        r_regress = cfg.delta * max(prev_layer - layer, 0) / cfg.max_depth

        expected = _expected_layer(idx, num_user_turns, cfg.max_depth)
        r_fast = cfg.stage_penalty * max(layer - expected, 0) / cfg.max_depth

        window = max(cfg.rep_window, 1)
        recent = layer_history[max(0, idx - window) : idx]
        rep_count = sum(1 for entry in recent if int(entry["layer"]) == layer)
        r_rep = cfg.delta_rep * (rep_count / window)

        r_raw = r_base + r_prog - r_regress - r_fast - r_rep
        r_k = _sigmoid(cfg.k2 * (r_raw - cfg.c2))
        rewards.append(r_k)

    r_episode = sum(rewards) / len(rewards)

    if cfg.bonus_depth and deepest_layer >= cfg.max_depth:
        r_episode += cfg.bonus_depth

    if cfg.bonus_progress and progress_count >= cfg.bonus_progress_min:
        r_episode += cfg.bonus_progress

    if terminate_reason == "empty_next_user":
        r_episode -= cfg.penalty_empty_user
    if terminate_reason == "no_progress_5":
        r_episode -= cfg.penalty_early_terminate

    return float(r_episode)


def _build_reward_cfg(default_cfg: RewardConfig, override: Optional[Dict[str, Any]]) -> RewardConfig:
    cfg = RewardConfig()
    for key, value in default_cfg.__dict__.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    if override:
        for key, value in override.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
    return cfg


def trl_reward_func(
    prompts,
    completions,
    completion_ids,
    trainer_state=None,
    **kwargs,
) -> List[float]:
    layer_history_batch = kwargs.get("layer_history")
    num_user_turns_batch = kwargs.get("num_user_turns")
    terminate_reason_batch = kwargs.get("terminate_reason")
    reward_config_batch = kwargs.get("reward_config")
    reward_defaults_batch = kwargs.get("reward_defaults")

    if layer_history_batch is None or num_user_turns_batch is None:
        return [0.0 for _ in completions]

    if len(layer_history_batch) != len(num_user_turns_batch) or len(layer_history_batch) != len(
        completions
    ):
        raise ValueError(
            "Mismatch in rollout metadata lengths: "
            f"len(layer_history)={len(layer_history_batch)}, "
            f"len(num_user_turns)={len(num_user_turns_batch)}, "
            f"len(completions)={len(completions)}."
        )

    rewards: List[float] = []
    for idx, (layer_history, num_user_turns) in enumerate(
        zip(layer_history_batch, num_user_turns_batch)
    ):
        defaults = RewardConfig()
        if reward_defaults_batch and idx < len(reward_defaults_batch):
            defaults = _build_reward_cfg(defaults, reward_defaults_batch[idx])

        overrides = None
        if reward_config_batch and idx < len(reward_config_batch):
            overrides = reward_config_batch[idx]

        cfg = _build_reward_cfg(defaults, overrides)

        terminate_reason = None
        if terminate_reason_batch and idx < len(terminate_reason_batch):
            terminate_reason = terminate_reason_batch[idx]

        reward = compute_episode_reward(
            layer_history=layer_history,
            num_user_turns=int(num_user_turns),
            cfg=cfg,
            terminate_reason=terminate_reason,
        )
        rewards.append(float(reward))

    return rewards
