"""Teacher-forcing evaluation for the user-agent model."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from typing import Any, Dict, List, Tuple

from src.models.infer_user_agent import generate_next, load_user_agent


def _extract_pairs(sample: Dict[str, Any]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    dialogue = sample.get("dialogue", [])
    if not isinstance(dialogue, list):
        return pairs

    for round_item in dialogue:
        if not isinstance(round_item, dict):
            continue
        turns = round_item.get("turns", [])
        if not isinstance(turns, list):
            continue
        speaker_text = None
        listener_text = None
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            role = turn.get("role")
            text = (turn.get("text") or "").strip()
            if not text:
                continue
            if role == "Speaker" and speaker_text is None:
                speaker_text = text
            elif role == "Listener" and listener_text is None:
                listener_text = text
        if speaker_text and listener_text:
            pairs.append((speaker_text, listener_text))
    return pairs


def _build_history(pairs: List[Tuple[str, str]], upto_index: int) -> List[Dict[str, str]]:
    history: List[Dict[str, str]] = []
    for speaker_text, listener_text in pairs[:upto_index]:
        history.append({"role": "Speaker", "text": speaker_text})
        history.append({"role": "Listener", "text": listener_text})
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Teacher-forcing evaluation for user-agent.")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--data_json", required=True)
    parser.add_argument("--output_json")
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--max_steps_per_dialogue", type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "npu"])
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    model, tokenizer, _ = load_user_agent(
        args.base_model,
        args.adapter_dir,
        device=args.device,
        local_files_only=True,
    )

    with open(args.data_json, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("data_json must be a top-level list")

    if args.output_json:
        output_path = args.output_json
    else:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = f"outputs/user_agent_eval/{timestamp}/predictions.json"

    results: List[Dict[str, Any]] = []
    processed = 0
    steps_generated = 0

    for sample in data:
        if not isinstance(sample, dict):
            continue
        dia_id = sample.get("dia_id")
        background = sample.get("first_explanation", "")
        pairs = _extract_pairs(sample)
        if not pairs:
            continue
        total_steps = len(pairs)
        if args.max_steps_per_dialogue is not None:
            total_steps = min(total_steps, args.max_steps_per_dialogue)

        for step_index in range(total_steps):
            history = _build_history(pairs, step_index)
            generated = generate_next(
                model,
                tokenizer,
                background,
                history,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=True,
            )
            gold = pairs[step_index][0]
            results.append(
                {
                    "dia_id": dia_id,
                    "step": step_index + 1,
                    "generated": generated,
                    "gold": gold,
                    "context_len": step_index,
                }
            )
            steps_generated += 1

        processed += 1
        if args.max_samples is not None and processed >= args.max_samples:
            break

    output_dir = output_path.rsplit("/", 1)[0]
    if output_dir:
        import os

        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)

    print(
        "Dialogues processed:",
        processed,
        "Steps generated:",
        steps_generated,
        "Output:",
        output_path,
    )


if __name__ == "__main__":
    main()
