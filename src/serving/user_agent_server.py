"""User-agent server: vLLM client + classifier on same NPU set."""

from __future__ import annotations

import argparse
import json
import time
from urllib.parse import urlparse
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List

from src.models.iceberg_classifier import IcebergClassifier
from src.serving.vllm_client import VLLMChatClient


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/classify":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            payload = json.loads(body)
            text = payload.get("text", "")

            layer, label, confidence = self.server.classifier.predict(text)
            resp = {
                "layer": layer,
                "label": label,
                "confidence": confidence,
            }

            resp_bytes = json.dumps(resp).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_bytes)))
            self.end_headers()
            self.wfile.write(resp_bytes)
            return

        if path != "/generate":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        payload = json.loads(body)
        messages = payload.get("messages", [])

        for attempt in range(3):
            try:
                result = self.server.vllm.generate(
                    messages=messages,
                    max_new_tokens=self.server.max_new_tokens,
                    temperature=self.server.temperature,
                    top_p=self.server.top_p,
                    logprobs=False,
                )
                break
            except Exception as e:
                print(f"[sidecar] vLLM failed ({attempt+1}/3): {e}")
                time.sleep(0.5 * (attempt + 1))

        user_text = (result.get("text") or "").strip()

        layer, label, confidence = self.server.classifier.predict(user_text)

        resp = {
            "text": user_text,
            "layer": layer,
            "label": label,
            "confidence": confidence,
        }

        resp_bytes = json.dumps(resp).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp_bytes)))
        self.end_headers()
        self.wfile.write(resp_bytes)

    def log_message(self, format, *args):
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9102)
    parser.add_argument("--vllm_url", required=True)
    parser.add_argument("--vllm_model", required=True)
    parser.add_argument("--classifier_dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    vllm = VLLMChatClient(args.vllm_url, args.vllm_model)
    classifier = IcebergClassifier(args.classifier_dir, device=args.device, local_files_only=True)

    server = HTTPServer((args.host, args.port), Handler)
    server.vllm = vllm
    server.classifier = classifier
    server.max_new_tokens = args.max_new_tokens
    server.temperature = args.temperature
    server.top_p = args.top_p

    print(f"User-agent server listening on {args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
