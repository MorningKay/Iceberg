#!/usr/bin/env python3
"""Convert Iceberg dataset to RL prompt JSON for GRPO training."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

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

def main() -> None:
    parser = argparse.ArgumentParser(description="Convert dataset to RL prompt JSON.")
    parser.add_argument("--input_json", required=True, help="Path to source dataset JSON")
    parser.add_argument("--output_json", required=True, help="Path to output JSON")
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--reward_config_json")
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("input_json must be a top-level list")

    output: List[Dict[str, Any]] = []
    processed = 0
    skipped = 0

    for sample in data:
        if not isinstance(sample, dict):
            skipped += 1
            continue
        prompt_text = sample.get("dialogue")[0].get("turns")[0].get("text")
        prompt = [
            {"role": "system", "content": S_MAIN},
            {"role": "user", "content": prompt_text},
        ]
        first_explanation = sample.get("first_explanation")
        dia_id = sample.get("dia_id")
        if not prompt or not first_explanation or not dia_id:
            skipped += 1
            continue
        output.append(
            {
                "prompt": prompt,
                "first_explanation": str(first_explanation),
                "dia_id": str(dia_id),
            }
        )
        processed += 1
        if args.max_samples is not None and processed >= args.max_samples:
            break

    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)

    print(
        "Converted samples:",
        processed,
        "Skipped:",
        skipped,
        "Output:",
        args.output_json,
    )


if __name__ == "__main__":
    main()
