#!/usr/bin/env python3
"""Convert Iceberg dialogue dataset to ShareGPT JSON for user-agent SFT."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional, Tuple

SYSTEM_PROMPT = (
    "我最近有点不顺，我想找人聊聊，只说我自己的真实感受和想法，不分析别人也不给建议。"
)


def _extract_round_texts(round_item: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    listener_text = None
    speaker_text = None
    turns = round_item.get("turns", [])
    if not isinstance(turns, list):
        return None, None
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        text = (turn.get("text") or "").strip()
        if not text:
            continue
        if role == "Listener" and listener_text is None:
            listener_text = text
        elif role == "Speaker" and speaker_text is None:
            speaker_text = text
    return listener_text, speaker_text


def _build_conversations(sample: Dict[str, Any], background_field: str) -> Tuple[List[Dict[str, str]], int]:
    background = (sample.get(background_field) or "").strip()
    conversations: List[Dict[str, str]] = [
        {"from": "system", "value": SYSTEM_PROMPT},
        {"from": "user", "value": f"<INIT>\n{background}"},
    ]

    skipped_rounds = 0
    dialogue = sample.get("dialogue", [])
    if not isinstance(dialogue, list):
        return conversations, 0

    for round_item in dialogue:
        if not isinstance(round_item, dict):
            skipped_rounds += 1
            continue
        listener_text, speaker_text = _extract_round_texts(round_item)
        if not listener_text or not speaker_text:
            skipped_rounds += 1
            continue
        conversations.append({"from": "user", "value": listener_text})
        conversations.append({"from": "assistant", "value": speaker_text})

    return conversations, skipped_rounds


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Iceberg dataset to ShareGPT JSON.")
    parser.add_argument("--input_json", required=True, help="Path to source dataset JSON")
    parser.add_argument("--output_json", required=True, help="Path to output ShareGPT JSON")
    parser.add_argument("--background_field", default="first_explanation")
    parser.add_argument("--max_samples", type=int)
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a top-level list")

    output: List[Dict[str, Any]] = []
    total_rounds = 0
    skipped_rounds = 0
    processed_samples = 0

    for sample in data:
        if not isinstance(sample, dict):
            continue
        conversations, skipped = _build_conversations(sample, args.background_field)
        output.append({"conversations": conversations})
        processed_samples += 1
        if isinstance(sample.get("dialogue"), list):
            total_rounds += len(sample["dialogue"])
        skipped_rounds += skipped
        if args.max_samples is not None and processed_samples >= args.max_samples:
            break

    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)

    print(
        "Converted samples:",
        processed_samples,
        "Total rounds:",
        total_rounds,
        "Skipped rounds:",
        skipped_rounds,
        "Output:",
        args.output_json,
    )


if __name__ == "__main__":
    main()
