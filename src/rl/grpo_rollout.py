"""Custom multi-turn rollout for GRPO training."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer

from src.models.infer_user_agent import build_init_message, generate_next, load_user_agent
from src.models.iceberg_classifier import IcebergClassifier

S_MAIN = (
    "你是一名同理心对话代理。目标是在安全前提下，帮助来访者被理解与安顿情绪，并引导从外显到内在的逐级探索。"
    "先判断用户目前所在层级，但不要外在解释，之后用1-2句共情性反映，每轮逐步向深下潜，"
    "优先安全与关系感，而非求快求全。要简洁温和，无批判和说教。\n\n"
    "冰山层级（由外到内，判定以用户“最新一条发言”为准）：\n"
    "- behavior（行为）：可直接观察的言行/反应\n"
    "- coping（应对）：为维护自我价值的防御/姿态（讨好、指责、过度理性、打岔等）\n"
    "- feelings（感受）：一阶情绪与身体线索（紧张、酸胀、委屈等）\n"
    "- feelings about feelings（对情绪的情绪）：对前述情绪的二阶反应（因生气而羞愧等）\n"
    "- perceptions（知觉/意义）：对事件的解释/归因（区分事实 vs 解释）\n"
)

S_USER = "我最近有点不顺，接下来我想找人聊聊，只说我自己的真实感受和想法，不分析别人也不给建议。"


@dataclass
class RolloutConfig:
    max_turns: int = 20
    user_max_new_tokens: int = 128
    user_temperature: float = 0.7
    user_top_p: float = 0.9
    user_do_sample: bool = True
    main_max_new_tokens: int = 128
    main_temperature: float = 0.7
    main_top_p: float = 0.9
    main_do_sample: bool = True
    return_texts: bool = False


@dataclass
class RolloutState:
    layer_history: List[Dict[str, Any]] = field(default_factory=list)
    terminate_reason: str = ""
    num_user_turns: int = 1


def _device_from_model(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def _apply_chat_template(tokenizer, messages: List[Dict[str, str]]) -> torch.Tensor:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    text = "\n".join(f"[{m['role']}] {m['content']}" for m in messages) + "\n[assistant]"
    return tokenizer(text, return_tensors="pt").input_ids


def _generate_main(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, str]],
    cfg: RolloutConfig,
) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
    prompt_ids = _apply_chat_template(tokenizer, messages)
    prompt_ids = prompt_ids.to(_device_from_model(model))

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=prompt_ids,
            max_new_tokens=cfg.main_max_new_tokens,
            temperature=cfg.main_temperature,
            top_p=cfg.main_top_p,
            do_sample=cfg.main_do_sample,
        )

    generated_ids = output_ids[0][prompt_ids.shape[-1] :]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    with torch.no_grad():
        outputs = model(
            input_ids=output_ids[:, :-1],
        )
        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)
        token_logprobs = log_probs.gather(
            -1, output_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        generated_logprobs = token_logprobs[0][prompt_ids.shape[-1] - 1 :]

    return text, generated_ids, generated_logprobs, prompt_ids


def _layer_unchanged_for_5(layer_history: List[Dict[str, Any]]) -> bool:
    if len(layer_history) < 5:
        return False
    last = layer_history[-1]["layer"]
    return all(entry["layer"] == last for entry in layer_history[-5:])


def _build_user_history_from_h_user(h_user: List[Dict[str, str]]) -> List[Dict[str, str]]:
    history: List[Dict[str, str]] = []
    for item in h_user[2:]:
        role = item.get("role")
        content = item.get("content", "")
        if not content:
            continue
        if role == "assistant":
            history.append({"role": "Speaker", "text": content})
        elif role == "user":
            history.append({"role": "Listener", "text": content})
    return history


def run_episode(
    prompt: str,
    first_explanation: str,
    main_model: torch.nn.Module,
    main_tokenizer: AutoTokenizer,
    user_agent_bundle: Tuple[torch.nn.Module, AutoTokenizer, str],
    classifier: IcebergClassifier,
    cfg: RolloutConfig,
    reward_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    h_main = [
        {"role": "system", "content": S_MAIN},
        {"role": "user", "content": prompt},
    ]
    h_user = [
        {"role": "system", "content": S_USER},
        {"role": "user", "content": build_init_message(first_explanation)},
        {"role": "assistant", "content": prompt},
    ]

    state = RolloutState()
    layer, label, conf = classifier.predict(prompt)
    state.layer_history.append(
        {"turn": 1, "layer": layer, "label": label, "confidence": conf}
    )

    completion_ids: List[int] = []
    logprobs: List[float] = []
    assistant_texts: List[str] = []
    user_texts: List[str] = [prompt]
    prompt_ids: Optional[List[int]] = None

    for t in range(1, cfg.max_turns + 1):
        assistant_text, gen_ids, gen_logprobs, prompt_tensor = _generate_main(
            main_model, main_tokenizer, h_main, cfg
        )
        if prompt_ids is None:
            prompt_ids = prompt_tensor[0].tolist()
        if assistant_text:
            assistant_texts.append(assistant_text)

        h_main.append({"role": "assistant", "content": assistant_text})
        h_user.append({"role": "user", "content": assistant_text})

        completion_ids.extend(gen_ids.tolist())
        logprobs.extend(gen_logprobs.tolist())

        if t == cfg.max_turns:
            state.terminate_reason = "max_turns"
            break
        if _layer_unchanged_for_5(state.layer_history):
            state.terminate_reason = "no_progress_5"
            break

        user_model, user_tokenizer, _ = user_agent_bundle
        user_text = ""
        for _ in range(3):
            user_text = generate_next(
                user_model,
                user_tokenizer,
                first_explanation,
                _build_user_history_from_h_user(h_user),
                max_new_tokens=cfg.user_max_new_tokens,
                temperature=cfg.user_temperature,
                top_p=cfg.user_top_p,
                do_sample=cfg.user_do_sample,
            )
            if user_text:
                break
        if not user_text:
            state.terminate_reason = "empty_next_user"
            break

        user_texts.append(user_text)
        layer, label, conf = classifier.predict(user_text)
        state.layer_history.append(
            {
                "turn": len(state.layer_history) + 1,
                "layer": layer,
                "label": label,
                "confidence": conf,
            }
        )
        h_main.append({"role": "user", "content": user_text})
        h_user.append({"role": "assistant", "content": user_text})
        state.num_user_turns += 1

    result = {
        "prompt_ids": [prompt_ids if prompt_ids is not None else []],
        "completion_ids": [completion_ids],
        "logprobs": [logprobs],
        "layer_history": [state.layer_history],
        "terminate_reason": [state.terminate_reason],
        "num_user_turns": [state.num_user_turns],
        "reward_config": [reward_config],
    }
    if cfg.return_texts:
        result["assistant_texts"] = [assistant_texts]
        result["user_texts"] = [user_texts]
    return result
