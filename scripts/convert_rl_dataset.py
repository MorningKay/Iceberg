#!/usr/bin/env python3
"""Convert Iceberg dataset to RL prompt JSON for GRPO training."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert dataset to RL prompt JSON.")
    parser.add_argument("--input_json", required=True, help="Path to source dataset JSON")
    parser.add_argument("--output_json", required=True, help="Path to output JSON")
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--reward_config_json")
    args = parser.parse_args()

    reward_config: Dict[str, Any] = {}
    if args.reward_config_json:
        try:
            reward_config = json.loads(args.reward_config_json)
        except json.JSONDecodeError as exc:
            raise ValueError("reward_config_json must be valid JSON") from exc
        if not isinstance(reward_config, dict):
            raise ValueError("reward_config_json must decode to a JSON object")

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
        seed = sample.get("dialogue")[0].get("turns")[0].get("text")
        first_explanation = sample.get("first_explanation")
        dia_id = sample.get("dia_id")
        if not seed or not first_explanation or not dia_id:
            skipped += 1
            continue
        output.append(
            {
                "prompt": str(seed),
                "first_explanation": str(first_explanation),
                "dia_id": str(dia_id),
                "reward_config": dict(reward_config),
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
