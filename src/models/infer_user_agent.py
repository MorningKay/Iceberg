"""Inference helper for the user-agent (Speaker simulator) model."""

from __future__ import annotations

from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = (
    "我最近有点不顺，我想找人聊聊，只说我自己的真实感受和想法，不分析别人也不给建议。"
)


def build_init_message(background: str) -> str:
    return f"<INIT>\n{background}"


def _resolve_device(device: str) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if device not in {"auto", "npu"}:
        raise ValueError("device must be one of: auto, cpu, npu")

    try:
        import torch_npu  # type: ignore

        _ = torch_npu
    except Exception as exc:
        raise RuntimeError(
            "torch_npu is required for NPU inference. Use device='cpu' or run in the Ascend NPU environment."
        ) from exc

    if not hasattr(torch, "npu"):
        raise RuntimeError("torch.npu is not available in this environment")

    torch.npu.set_device(0)
    return torch.device("npu:0")


def load_user_agent(model_dir: str, device: str = "auto") -> Dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)

    resolved = _resolve_device(device)
    model.to(resolved)
    model.eval()
    return {"tokenizer": tokenizer, "model": model, "device": resolved}


def _format_history(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    for item in history:
        role = item.get("role")
        text = (item.get("text") or "").strip()
        if not text:
            continue
        if role == "Listener":
            messages.append({"role": "user", "content": text})
        elif role == "Speaker":
            messages.append({"role": "assistant", "content": text})
    return messages


def generate_next(
    model_bundle: Dict[str, Any],
    background: str,
    history: List[Dict[str, str]],
    **gen_kwargs,
) -> str:
    tokenizer = model_bundle["tokenizer"]
    model = model_bundle["model"]
    device = model_bundle["device"]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_init_message(background)},
    ]
    messages.extend(_format_history(history))

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            **gen_kwargs,
        )

    generated = output_ids[0][input_ids.shape[-1] :]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text.strip()
