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


_EARLY_STAGE_PROFILE = [1.00, 1.00, 0.55, 0.20, 0.05]
_MIDDLE_STAGE_PROFILE = [0.35, 0.75, 1.00, 1.00, 0.55]
_LATE_STAGE_PROFILE = [0.10, 0.30, 0.70, 1.00, 1.00]
_DEFAULT_TERM_NAMES = ["stage_match", "transition", "stability", "coverage"]


def _empty_reward_extra(num_items: int, term_names: Optional[List[str]] = None) -> Dict[str, Any]:
    names = term_names or _DEFAULT_TERM_NAMES
    reward_terms = {name: [0.0 for _ in range(num_items)] for name in names}
    reward_term_means = {name: 0.0 for name in names}
    return {
        "reward_terms": reward_terms,
        "reward_term_means": reward_term_means,
    }


def _blend_profiles(left: List[float], right: List[float], t: float) -> List[float]:
    blend = max(0.0, min(1.0, t))
    return [(1.0 - blend) * l + blend * r for l, r in zip(left, right)]


def _stage_profile_weights(u: float, cfg: RewardConfig) -> List[float]:
    s = _sigmoid(cfg.kappa * (u - cfg.c))
    if s <= 0.5:
        return _blend_profiles(_EARLY_STAGE_PROFILE, _MIDDLE_STAGE_PROFILE, s / 0.5)
    return _blend_profiles(_MIDDLE_STAGE_PROFILE, _LATE_STAGE_PROFILE, (s - 0.5) / 0.5)


def _transition_base_score(delta: int) -> float:
    if delta == 1:
        return 1.0
    if delta == 0:
        return 0.85
    if delta == 2:
        return 0.60
    if delta == -1:
        return 0.55
    if delta == -2:
        return 0.25
    return 0.0


def _max_confidence_for_layers(layer_history: List[Dict[str, Any]], valid_layers: set[int]) -> float:
    best = 0.0
    for item in layer_history:
        layer = int(item["layer"])
        if layer in valid_layers:
            best = max(best, float(item["confidence"]))
    return best


def compute_episode_reward(
    layer_history: List[Dict[str, Any]],
    num_user_turns: int,
    cfg: RewardConfig,
    terminate_reason: Optional[str] = None,
) -> Dict[str, Any]:
    if num_user_turns <= 0 or not layer_history:
        return {
            "reward": 0.0,
            "reward_terms": {
                "stage_match": 0.0,
                "transition": 0.0,
                "stability": 0.0,
                "coverage": 0.0,
            },
        }

    stage_scores: List[float] = []
    transition_scores: List[float] = []
    layers: List[int] = []
    confidences: List[float] = []
    prev_layer: Optional[int] = None
    prev_confidence: Optional[float] = None

    for idx, item in enumerate(layer_history, start=1):
        layer = max(1, min(5, int(item["layer"])))
        confidence = max(0.0, min(1.0, float(item["confidence"])))
        layers.append(layer)
        confidences.append(confidence)
        u = idx / float(num_user_turns)
        stage_weights = _stage_profile_weights(u, cfg)
        stage_scores.append(confidence * stage_weights[layer - 1])

        if prev_layer is None or prev_confidence is None:
            transition_scores.append(0.60 * confidence)
        else:
            delta = layer - prev_layer
            pair_confidence = 0.5 * (prev_confidence + confidence)
            transition_scores.append(_transition_base_score(delta) * pair_confidence)

        prev_layer = layer
        prev_confidence = confidence

    stage_match = float(sum(stage_scores) / len(stage_scores)) if stage_scores else 0.0
    transition = float(sum(transition_scores) / len(transition_scores)) if transition_scores else 0.0

    tail_n = max(2, math.ceil(num_user_turns / 3))
    tail_layers = layers[-tail_n:]
    tail_confidences = confidences[-tail_n:]
    tail_deep_presence = (
        sum(conf * max(layer - 2, 0) / 3.0 for layer, conf in zip(tail_layers, tail_confidences))
        / len(tail_layers)
    )
    if len(tail_layers) <= 1:
        tail_smoothness = 1.0
    else:
        tail_smoothness_scores = [
            1.0 if abs(curr - prev) <= 1 else 0.5 if abs(curr - prev) == 2 else 0.0
            for prev, curr in zip(tail_layers, tail_layers[1:])
        ]
        tail_smoothness = sum(tail_smoothness_scores) / len(tail_smoothness_scores)
    stability = float(0.70 * tail_deep_presence + 0.30 * tail_smoothness)

    surface_conf = _max_confidence_for_layers(layer_history, {1, 2})
    emotion_conf = _max_confidence_for_layers(layer_history, {3, 4})
    deep_conf = _max_confidence_for_layers(layer_history, {5})
    coverage = float((surface_conf + emotion_conf + deep_conf) / 3.0)

    reward_terms = {
        "stage_match": stage_match,
        "transition": transition,
        "stability": stability,
        "coverage": coverage,
    }
    r_episode = (
        0.35 * reward_terms["stage_match"]
        + 0.30 * reward_terms["transition"]
        + 0.20 * reward_terms["stability"]
        + 0.15 * reward_terms["coverage"]
    )

    if terminate_reason == "empty_next_user":
        r_episode -= cfg.penalty_empty_user
    if terminate_reason == "no_progress_5":
        r_episode -= cfg.penalty_early_terminate

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
