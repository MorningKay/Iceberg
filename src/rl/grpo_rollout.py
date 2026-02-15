"""Custom multi-turn rollout for GRPO training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer

from src.models.infer_user_agent import build_init_message
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


def _apply_chat_template(tokenizer, messages: List[Dict[str, str]]):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )


def _generate_main(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, str]],
    cfg: RolloutConfig,
) -> Tuple[str, List[int], List[float], List[int]]:
    was_training = model.training
    model.eval()
    try:
        enc = _apply_chat_template(tokenizer, messages)
        if hasattr(enc, "to"):
            enc = enc.to(_device_from_model(model))
        else:
            enc = {k: v.to(_device_from_model(model)) for k, v in enc.items()}

        assert "input_ids" in enc
        input_ids = enc["input_ids"]
        assert torch.is_tensor(input_ids) and input_ids.ndim == 2
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            assert attention_mask.shape == input_ids.shape

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=cfg.main_max_new_tokens,
                temperature=cfg.main_temperature,
                top_p=cfg.main_top_p,
                do_sample=cfg.main_do_sample,
            )

        generated_ids = output_ids[0][input_ids.shape[-1] :]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        with torch.inference_mode():
            outputs = model(input_ids=output_ids[:, :-1], attention_mask=None)
            logits = outputs.logits
            log_probs = torch.log_softmax(logits, dim=-1)
            token_logprobs = log_probs.gather(
                -1, output_ids[:, 1:].unsqueeze(-1)
            ).squeeze(-1)
            generated_logprobs = token_logprobs[0][input_ids.shape[-1] - 1 :]
    finally:
        model.train(was_training)

    return (
        text,
        generated_ids.detach().cpu().tolist(),
        generated_logprobs.detach().cpu().tolist(),
        input_ids[0].detach().cpu().tolist(),
    )


def _layer_unchanged_for_5(layer_history: List[Dict[str, Any]]) -> bool:
    if len(layer_history) < 5:
        return False
    last = layer_history[-1]["layer"]
    return all(entry["layer"] == last for entry in layer_history[-5:])


def _generate_user(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, str]],
    cfg: RolloutConfig,
) -> str:
    was_training = model.training
    model.eval()
    try:
        enc = _apply_chat_template(tokenizer, messages)
        if hasattr(enc, "to"):
            enc = enc.to(_device_from_model(model))
        else:
            enc = {k: v.to(_device_from_model(model)) for k, v in enc.items()}

        assert "input_ids" in enc
        input_ids = enc["input_ids"]
        assert torch.is_tensor(input_ids) and input_ids.ndim == 2
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            assert attention_mask.shape == input_ids.shape

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=cfg.user_max_new_tokens,
                temperature=cfg.user_temperature,
                top_p=cfg.user_top_p,
                do_sample=cfg.user_do_sample,
            )

        generated_ids = output_ids[0][input_ids.shape[-1] :]
    finally:
        model.train(was_training)

    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def _extract_prompt_messages(prompt: Any) -> Tuple[List[Dict[str, str]], str, str]:
    if isinstance(prompt, list):
        messages = [
            {"role": m.get("role", ""), "content": m.get("content", "")}
            for m in prompt
            if isinstance(m, dict)
        ]
    else:
        messages = [{"role": "user", "content": str(prompt)}]

    system_msg = ""
    user_msg = ""
    for msg in messages:
        if msg.get("role") == "system" and not system_msg:
            system_msg = msg.get("content", "")
        if msg.get("role") == "user":
            user_msg = msg.get("content", "")
    if not user_msg:
        user_msg = messages[-1].get("content", "") if messages else ""
    return messages, system_msg, user_msg


def run_episode(
    prompt_messages: List[Dict[str, str]],
    first_explanation: str,
    main_model: torch.nn.Module,
    main_tokenizer: AutoTokenizer,
    user_model: torch.nn.Module,
    user_tokenizer: AutoTokenizer,
    classifier: IcebergClassifier,
    cfg: RolloutConfig,
    assistant_client=None,
    user_client=None,
) -> Dict[str, Any]:
    _messages, system_msg, user_msg = _extract_prompt_messages(prompt_messages)
    system_msg = system_msg or S_MAIN

    h_main = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    h_user = [
        {"role": "system", "content": S_USER},
        {"role": "user", "content": build_init_message(first_explanation)},
        {"role": "assistant", "content": user_msg},
    ]

    state = RolloutState()
    layer, label, conf = classifier.predict(user_msg)
    state.layer_history.append(
        {"turn": 1, "layer": layer, "label": label, "confidence": conf}
    )

    completion_ids: List[int] = []
    logprobs: List[float] = []
    prompt_ids: Optional[List[int]] = None

    for t in range(1, cfg.max_turns + 1):
        if assistant_client is not None:
            result = assistant_client.generate(
                messages=h_main,
                max_new_tokens=cfg.main_max_new_tokens,
                temperature=cfg.main_temperature,
                top_p=cfg.main_top_p,
                logprobs=True,
            )
            assistant_text = (result.get("text") or "").strip()
            gen_ids = result.get("token_ids") or []
            gen_logprobs = result.get("token_logprobs") or []
            if prompt_ids is None:
                prompt_ids = _apply_chat_template(main_tokenizer, h_main)["input_ids"][0].tolist()
        else:
            assistant_text, gen_ids, gen_logprobs, step_prompt_ids = _generate_main(
                main_model, main_tokenizer, h_main, cfg
            )
            if prompt_ids is None:
                prompt_ids = step_prompt_ids

        h_main.append({"role": "assistant", "content": assistant_text})
        h_user.append({"role": "user", "content": assistant_text})

        completion_ids.extend(gen_ids)
        logprobs.extend(gen_logprobs)

        if t == cfg.max_turns:
            state.terminate_reason = "max_turns"
            break
        if _layer_unchanged_for_5(state.layer_history):
            state.terminate_reason = "no_progress_5"
            break

        user_text = ""
        for _ in range(3):
            if user_client is not None:
                resp = user_client.generate(h_user)
                user_text = (resp.get("text") or "").strip()
                layer = resp.get("layer")
                label = resp.get("label")
                conf = resp.get("confidence")
            else:
                user_text = _generate_user(user_model, user_tokenizer, h_user, cfg)
                layer = label = conf = None
            if user_text:
                break
        if not user_text:
            state.terminate_reason = "empty_next_user"
            break

        if layer is None or label is None or conf is None:
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

    return {
        "prompt_ids": prompt_ids or [],
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "layer_history": state.layer_history,
        "terminate_reason": state.terminate_reason,
        "num_user_turns": state.num_user_turns,
    }
