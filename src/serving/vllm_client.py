"""Simple vLLM OpenAI-compatible client for chat completions."""

from __future__ import annotations

import json
import urllib.request
from typing import Any, Dict, List


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
