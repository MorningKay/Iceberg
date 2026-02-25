"""Simple vLLM OpenAI-compatible client for chat completions."""

from __future__ import annotations

import json
import urllib.request
from typing import Any, Dict, List, Optional

import torch


class VLLMChatClient:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        logprobs: bool = True,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = 1

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))

        choice = body["choices"][0]
        text = choice["message"]["content"]
        token_ids = None
        token_logprobs = None
        logprobs_obj = choice.get("logprobs")
        if logprobs_obj:
            token_ids = logprobs_obj.get("token_ids")
            token_logprobs = logprobs_obj.get("token_logprobs")

        return {
            "text": text,
            "token_ids": token_ids,
            "token_logprobs": token_logprobs,
        }


class TRLVLLMServeClient:
    """Client for TRL `trl vllm-serve` server mode.

    TRL's server exposes POST /generate/ with JSON:
      {"prompts": ["..."], "max_tokens": int, ...}
    and returns token ids under "completion_ids".

    This client decodes token ids to text and can compute per-token logprobs
    locally using the provided main_model.
    """

    def __init__(self, base_url: str, tokenizer):
        self.base_url = base_url.rstrip("/")
        self.tokenizer = tokenizer

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        logprobs: bool = True,
        *,
        main_model: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        if not hasattr(self.tokenizer, "apply_chat_template"):
            raise RuntimeError("tokenizer.apply_chat_template is required for TRL vllm-serve client")

        prompt_str = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            return_tensors=None,
        )

        payload: Dict[str, Any] = {
            "prompts": [prompt_str],
            "max_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "n": 1,
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/generate/",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body_bytes = resp.read()
        except Exception as exc:
            raise RuntimeError(f"TRL vllm-serve request failed: {exc}") from exc

        try:
            body = json.loads(body_bytes.decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Failed to parse TRL vllm-serve response: {body_bytes[:200]!r}") from exc

        if "completion_ids" not in body:
            raise RuntimeError(f"TRL vllm-serve response missing completion_ids: keys={list(body.keys())}")

        token_ids = body["completion_ids"][0]
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()

        token_logprobs = None
        if logprobs:
            if main_model is None:
                raise RuntimeError("logprobs=True requires main_model for local logprob computation")

            prompt_enc = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            prompt_ids = prompt_enc["input_ids"][0]
            gen_ids = torch.tensor(token_ids, dtype=torch.long)
            input_ids = torch.cat([prompt_ids, gen_ids], dim=0).unsqueeze(0)

            if device is None:
                device = next(main_model.parameters()).device
            input_ids = input_ids.to(device)

            with torch.no_grad():
                outputs = main_model(input_ids=input_ids)
                logits = outputs.logits  # [1, seq_len, vocab]
                log_probs = torch.log_softmax(logits, dim=-1)

                # log p(x_t | x_<t>) for each token in input_ids[:, 1:]
                next_token_ids = input_ids[:, 1:]
                token_logp = log_probs[:, :-1, :].gather(-1, next_token_ids.unsqueeze(-1)).squeeze(-1)

                # Generated portion corresponds to positions prompt_len..prompt_len+gen_len-1
                prompt_len = int(prompt_ids.shape[0])
                gen_logp = token_logp[0][prompt_len - 1 : prompt_len - 1 + len(token_ids)]
                token_logprobs = gen_logp.detach().cpu().tolist()

        return {
            "text": text,
            "token_ids": token_ids,
            "token_logprobs": token_logprobs,
        }
