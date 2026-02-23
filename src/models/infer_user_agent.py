"""Inference helper for the user-agent (Speaker simulator) with LoRA adapter."""

from __future__ import annotations

import warnings
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = "我最近有点不顺，我想找人聊聊，只说我自己的真实感受和想法，不分析别人也不给建议。"


def build_init_message(background: str) -> str:
    background = (background or "").strip()
    return f"<INIT>\n{background}"


def _resolve_device(device: str) -> str:
    if device == "cpu":
        return "cpu"
    if device not in {"auto", "npu"}:
        raise ValueError("device must be one of: auto, cpu, npu")

    try:
        import torch_npu  # type: ignore

        _ = torch_npu
        return "npu"
    except Exception:
        if device == "npu":
            raise RuntimeError(
                "torch_npu is required for NPU inference. Use device='cpu' or run in the Ascend NPU environment."
            )
        return "cpu"


def load_user_agent(
    base_model: str,
    adapter_dir: str,
    device: str = "auto",
    local_files_only: bool = True,
) -> Tuple[torch.nn.Module, AutoTokenizer, str]:
    adapter_path = Path(adapter_dir)
    if not adapter_path.exists():
        raise FileNotFoundError(f"adapter_dir not found: {adapter_dir}")

    has_config = (adapter_path / "adapter_config.json").exists()
    has_weights = (adapter_path / "adapter_model.safetensors").exists() or (
        adapter_path / "adapter_model.bin"
    ).exists()
    if not has_config or not has_weights:
        raise FileNotFoundError(
            "adapter_dir must contain adapter_config.json and adapter_model.safetensors (or adapter_model.bin)"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        local_files_only=local_files_only,
        trust_remote_code=True,
    )

    from peft import PeftModel

    model = PeftModel.from_pretrained(
        base,
        adapter_dir,
        local_files_only=local_files_only,
    )

    resolved = _resolve_device(device)
    if resolved == "npu":
        try:
            import torch_npu  # type: ignore

            _ = torch_npu
        except Exception as exc:
            raise RuntimeError(
                "torch_npu is required for NPU inference. Use device='cpu' or run in the Ascend NPU environment."
            ) from exc
        if not hasattr(torch, "npu"):
            raise RuntimeError("torch.npu is not available in this environment")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.npu.set_device(local_rank)
        model.to(f"npu:{local_rank}")
    else:
        model.to("cpu")

    model.eval()
    return model, tokenizer, resolved


def build_messages(background: str, history: List[Dict]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_init_message(background)},
    ]

    for item in history:
        role = item.get("role")
        text = (item.get("text") or "").strip()
        if not text:
            continue
        if role == "Listener":
            messages.append({"role": "user", "content": text})
        elif role == "Speaker":
            messages.append({"role": "assistant", "content": text})
        else:
            warnings.warn(f"Unknown role '{role}' in history; skipping", RuntimeWarning)

    return messages


def _build_fallback_prompt(messages: List[Dict[str, str]]) -> str:
    parts = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        parts.append(f"[{role}] {content}")
    parts.append("[assistant]")
    return "\n".join(parts)


def generate_next(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    background: str,
    history: List[Dict],
    *,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> str:
    messages = build_messages(background, history)

    if hasattr(tokenizer, "apply_chat_template"):
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    else:
        prompt_text = _build_fallback_prompt(messages)
        prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids

    device = next(model.parameters()).device
    prompt_ids = prompt_ids.to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )

    generated = output_ids[0][prompt_ids.shape[-1] :]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text.strip()
