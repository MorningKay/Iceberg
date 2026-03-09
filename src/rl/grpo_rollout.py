"""Custom multi-turn rollout for GRPO training."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer

from src.models.infer_user_agent import build_init_message

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


def _apply_chat_template(
    tokenizer,
    messages: List[Dict[str, str]],
    *,
    tokenize: bool = True,
    add_generation_prompt: bool = True,
):
    if tokenize:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt",
        )
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        return_tensors=None,
    )


def _chat_template_ids(
    tokenizer,
    messages: List[Dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> List[int]:
    encoded = _apply_chat_template(
        tokenizer,
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    if isinstance(encoded, dict):
        input_ids = encoded["input_ids"]
    else:
        input_ids = encoded

    if torch.is_tensor(input_ids):
        if input_ids.ndim == 2:
            return input_ids[0].detach().cpu().tolist()
        if input_ids.ndim == 1:
            return input_ids.detach().cpu().tolist()
    if isinstance(input_ids, list):
        if input_ids and isinstance(input_ids[0], list):
            return [int(x) for x in input_ids[0]]
        return [int(x) for x in input_ids]
    raise TypeError(f"Unsupported chat template output type: {type(input_ids)}")


def _logprobs_for_suffix(
    model: torch.nn.Module,
    full_ids: List[int],
    suffix_len: int,
) -> List[float]:
    if suffix_len == 0:
        return []

    was_training = model.training
    model.eval()
    try:
        input_ids = torch.tensor(full_ids, dtype=torch.long, device=_device_from_model(model)).unsqueeze(0)
        with torch.inference_mode():
            outputs = model(input_ids=input_ids, attention_mask=None)
            logits = outputs.logits
            log_probs = torch.log_softmax(logits, dim=-1)
            token_logprobs = log_probs[:, :-1, :].gather(
                -1, input_ids[:, 1:].unsqueeze(-1)
            ).squeeze(-1)
    finally:
        model.train(was_training)

    if suffix_len > token_logprobs.shape[1]:
        raise ValueError(
            f"Requested suffix_len={suffix_len} exceeds available shifted token logprobs={token_logprobs.shape[1]}"
        )
    return token_logprobs[0][-suffix_len:].detach().cpu().tolist()


def _assert_prefix(
    current_full_ids: List[int],
    next_ids: List[int],
    *,
    turn: int,
    boundary: str,
) -> None:
    if next_ids[: len(current_full_ids)] == current_full_ids:
        return

    mismatch = None
    limit = min(len(current_full_ids), len(next_ids))
    for idx in range(limit):
        if current_full_ids[idx] != next_ids[idx]:
            mismatch = idx
            break
    if mismatch is None and len(current_full_ids) > len(next_ids):
        mismatch = len(next_ids)
    if mismatch is None:
        mismatch = 0
    window = 10
    a0 = max(0, mismatch - window)
    a1 = mismatch + window
    if os.environ.get("ICEBERG_DEBUG_ROLLOUT") == "1":
        print(
            "[ICEBERG_DEBUG_ROLLOUT] prefix_mismatch "
            f"boundary={boundary} turn={turn} mismatch_index={mismatch} "
            f"current_window={current_full_ids[a0:a1]} next_window={next_ids[a0:a1]}"
        )
    raise ValueError(
        f"TRL contiguity check failed at {boundary} boundary "
        "(current_full_ids must be a prefix of the canonical chat-template ids). "
        f"turn={turn} len(current_full_ids)={len(current_full_ids)} len(next_ids)={len(next_ids)} "
        f"mismatch_index={mismatch} current_full_ids_window={current_full_ids[a0:a1]} "
        f"next_ids_window={next_ids[a0:a1]}"
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


def _extract_prompt_messages(messages: List[Dict[str, str]]) -> Tuple[str, str]:
    system_msg = None
    last_user = None
    for m in messages:
        if m["role"] == "system":
            system_msg = m["content"]
        elif m["role"] == "user":
            last_user = m["content"]
    if last_user is None:
        raise ValueError("No user message found in parsed prompt.")
    return system_msg, last_user


def run_episode(
    prompt_messages: List[Dict[str, str]],
    first_explanation: str,
    main_model: torch.nn.Module,
    main_tokenizer: AutoTokenizer,
    user_model: Optional[torch.nn.Module],
    user_tokenizer: Optional[AutoTokenizer],
    classifier: Optional[object],
    cfg: RolloutConfig,
    assistant_client=None,
    user_client=None,
) -> Dict[str, Any]:
    system_msg, user_msg = _extract_prompt_messages(prompt_messages)
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
    if user_client is not None:
        resp = user_client.classify(user_msg)
        layer = resp.get("layer")
        label = resp.get("label")
        conf = resp.get("confidence")
    elif classifier is not None:
        layer, label, conf = classifier.predict(user_msg)
    else:
        raise RuntimeError(
            "Seed classification requires either user_client (preferred) or a local classifier."
        )
    if layer is None or label is None or conf is None:
        raise RuntimeError(
            "Seed user utterance classification is missing. Ensure /classify returns {layer,label,confidence}."
        )
    state.layer_history.append(
        {"turn": 1, "layer": layer, "label": label, "confidence": conf}
    )

    # The initial assistant generation prompt is the fixed prefix for the whole
    # trajectory. Every later token is derived from canonical chat-template
    # deltas so that completion_ids remains the exact contiguous suffix after
    # prompt_ids.
    prompt_ids = _chat_template_ids(
        main_tokenizer,
        h_main,
        add_generation_prompt=True,
    )

    completion_ids: List[int] = []
    assistant_token_mask: List[int] = []
    logprobs: List[float] = []
    # Optional debug metadata: offsets into completion_ids for each segment.
    turn_offsets: List[Dict[str, Any]] = []

    for t in range(1, cfg.max_turns + 1):
        if assistant_client is not None:
            result = assistant_client.generate(
                messages=h_main,
                max_new_tokens=cfg.main_max_new_tokens,
                temperature=cfg.main_temperature,
                top_p=cfg.main_top_p,
                logprobs=False,
            )
            assistant_text = (result.get("text") or "").strip()
        else:
            assistant_text, _gen_ids, _gen_logprobs, _step_prompt_ids = _generate_main(
                main_model, main_tokenizer, h_main, cfg
            )

        h_main.append({"role": "assistant", "content": assistant_text})
        h_user.append({"role": "user", "content": assistant_text})

        # Canonicalize the assistant action by diffing successive chat-template
        # tokenizations of h_main. This keeps completion_ids equal to the exact
        # suffix produced by the same tokenizer/template used for later turns.
        current_full_ids = prompt_ids + completion_ids
        after_assistant_ids = _chat_template_ids(
            main_tokenizer,
            h_main,
            add_generation_prompt=False,
        )
        _assert_prefix(
            current_full_ids,
            after_assistant_ids,
            turn=t,
            boundary="assistant",
        )
        assistant_delta_ids = after_assistant_ids[len(current_full_ids) :]

        start = len(completion_ids)
        completion_ids.extend(int(x) for x in assistant_delta_ids)
        assistant_token_mask.extend([1] * len(assistant_delta_ids))
        assistant_delta_logprobs = _logprobs_for_suffix(
            main_model,
            current_full_ids + assistant_delta_ids,
            len(assistant_delta_ids),
        )
        if len(assistant_delta_logprobs) != len(assistant_delta_ids):
            raise ValueError(
                f"Canonical assistant delta logprobs misaligned at turn {t}: "
                f"len(delta_ids)={len(assistant_delta_ids)} len(logprobs)={len(assistant_delta_logprobs)}"
            )
        logprobs.extend(float(x) for x in assistant_delta_logprobs)
        end = len(completion_ids)
        turn_offsets.append({"turn": t, "kind": "assistant", "start": start, "end": end})
        if os.environ.get("ICEBERG_DEBUG_ROLLOUT") == "1":
            print(
                "[ICEBERG_DEBUG_ROLLOUT] assistant_delta "
                f"turn={t} delta_len={len(assistant_delta_ids)} "
                f"completion_len={len(completion_ids)} logprobs_len={len(logprobs)} "
                f"mask_len={len(assistant_token_mask)}"
            )

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
                if user_model is None or user_tokenizer is None:
                    raise RuntimeError(
                        "Local user-agent generation requested but user_model/user_tokenizer is None."
                    )
                user_text = _generate_user(user_model, user_tokenizer, h_user, cfg)
                layer = label = conf = None
            if user_text:
                break
        if not user_text:
            state.terminate_reason = "empty_next_user"
            break

        if layer is None or label is None or conf is None:
            if classifier is None:
                raise RuntimeError(
                    "Missing (layer,label,confidence) for user_text and no local classifier is available."
                )
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

        current_full_ids = prompt_ids + completion_ids
        next_prompt_ids = _chat_template_ids(
            main_tokenizer,
            h_main,
            add_generation_prompt=True,
        )
        _assert_prefix(
            current_full_ids,
            next_prompt_ids,
            turn=t,
            boundary="user",
        )

        # User/environment/template continuation tokens stay in the trajectory
        # so prompt_ids + completion_ids remains contiguous, but they are masked
        # out of the policy loss and use dummy dense logprobs.
        delta_ids = next_prompt_ids[len(current_full_ids) :]
        if delta_ids:
            start = len(completion_ids)
            completion_ids.extend(int(x) for x in delta_ids)
            assistant_token_mask.extend([0] * len(delta_ids))
            logprobs.extend([0.0] * len(delta_ids))
            end = len(completion_ids)
            turn_offsets.append({"turn": t, "kind": "env", "start": start, "end": end})

    if not (len(completion_ids) == len(logprobs) == len(assistant_token_mask)):
        raise ValueError(
            "Length mismatch at end of episode: "
            f"len(completion_ids)={len(completion_ids)} len(logprobs)={len(logprobs)} "
            f"len(assistant_token_mask)={len(assistant_token_mask)}"
        )
    if any(m not in (0, 1, True, False) for m in assistant_token_mask):
        raise ValueError("assistant_token_mask must contain only 0/1 values")
    assistant_token_mask = [1 if bool(m) else 0 for m in assistant_token_mask]
    logprobs = [float(x) for x in logprobs]

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "assistant_token_mask": assistant_token_mask,
        "turn_offsets": turn_offsets,
        "layer_history": state.layer_history,
        "terminate_reason": state.terminate_reason,
        "num_user_turns": state.num_user_turns,
    }
