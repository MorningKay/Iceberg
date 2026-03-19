"""Reward computation for multi-turn iceberg dialogue RL."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


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


def _empty_reward_extra(num_items: int, term_names: Optional[List[str]] = None) -> Dict[str, Any]:
    names = term_names or ["base", "prog"]
    reward_terms = {name: [0.0 for _ in range(num_items)] for name in names}
    reward_term_means = {name: 0.0 for name in names}
    return {
        "reward_terms": reward_terms,
        "reward_term_means": reward_term_means,
    }


def compute_episode_reward(
    layer_history: List[Dict[str, Any]],
    num_user_turns: int,
    cfg: RewardConfig,
    terminate_reason: Optional[str] = None,
) -> Dict[str, Any]:
    if num_user_turns <= 0:
        return {
            "reward": 0.0,
            "reward_terms": {
                "base": 0.0,
                "prog": 0.0,
            },
        }

    rewards: List[float] = []
    progress_count = 0
    deepest_layer = 0
    reward_term_sums: Dict[str, float] = {
        "base": 0.0,
        "prog": 0.0,
    }

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
        reward_term_sums["base"] += r_base

        r_prog = cfg.beta if layer == prev_layer + 1 else 0.0
        reward_term_sums["prog"] += r_prog
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

    reward_terms = {
        name: float(total / num_user_turns)
        for name, total in reward_term_sums.items()
    }
    return {
        "reward": float(r_episode),
        "reward_terms": reward_terms,
    }


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
) -> Tuple[List[float], Dict[str, Any]]:
    layer_history_batch = kwargs.get("layer_history")
    num_user_turns_batch = kwargs.get("num_user_turns")
    terminate_reason_batch = kwargs.get("terminate_reason")
    reward_config_batch = kwargs.get("reward_config")
    reward_defaults_batch = kwargs.get("reward_defaults")

    if layer_history_batch is None or num_user_turns_batch is None:
        zero_rewards = [0.0 for _ in completions]
        return zero_rewards, _empty_reward_extra(len(zero_rewards))

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
    episode_results: List[Dict[str, Any]] = []
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

        reward_result = compute_episode_reward(
            layer_history=layer_history,
            num_user_turns=int(num_user_turns),
            cfg=cfg,
            terminate_reason=terminate_reason,
        )
        rewards.append(float(reward_result["reward"]))
        episode_results.append(reward_result)

    term_names: List[str] = []
    for result in episode_results:
        for name in result.get("reward_terms", {}):
            if name not in term_names:
                term_names.append(name)

    if not term_names:
        return rewards, _empty_reward_extra(len(rewards))

    reward_terms = {
        name: [
            float(result.get("reward_terms", {}).get(name, 0.0))
            for result in episode_results
        ]
        for name in term_names
    }
    reward_term_means = {
        name: float(sum(values) / len(values)) if values else 0.0
        for name, values in reward_terms.items()
    }
    extra = {
        "reward_terms": reward_terms,
        "reward_term_means": reward_term_means,
    }
    return rewards, extra
