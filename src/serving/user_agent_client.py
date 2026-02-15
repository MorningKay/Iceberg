"""User-agent service client (vLLM + classifier sidecar)."""

from __future__ import annotations

import json
import urllib.request
from typing import Any, Dict, List


class UserAgentClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def generate(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        payload = {"messages": messages}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
