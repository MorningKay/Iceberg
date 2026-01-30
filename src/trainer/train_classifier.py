"""Train the Iceberg classifier according to docs/classifier.md."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import evaluate
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.data.iceberg_dataset import (
    ID2LABEL,
    LABEL2ID,
    SupervisedExample,
    load_supervised_rows,
    make_splits,
)


@dataclass
class TrainConfig:
    model_name_or_path: str = "hfl/chinese-bert-wwm-ext"
    max_length: int = 256
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.0
    logging_steps: int = 50
    save_steps: int = 500
    eval_steps: int = 500
    eval_strategy: str = "steps"
    seed: int = 42
    fp16: bool = False
    bf16: bool = False
    output_dir: str = "outputs/classifier/run"


def _load_yaml(path: str) -> Dict:
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _merge_config(cli_args: argparse.Namespace) -> TrainConfig:
    cfg = TrainConfig()
    if cli_args.config:
        data = _load_yaml(cli_args.config)
        train_data = data.get("train", {})
        for key, value in train_data.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        if "label2id" in data or "id2label" in data:
            label2id = data.get("label2id", {})
            id2label = data.get("id2label", {})
            if label2id and label2id != LABEL2ID:
                raise ValueError("label2id in config must match fixed LABEL2ID")
            if id2label and id2label != {str(k): v for k, v in ID2LABEL.items()}:
                raise ValueError("id2label in config must match fixed ID2LABEL")
    for key in [
        "model_name_or_path",
        "max_length",
        "learning_rate",
        "weight_decay",
        "num_train_epochs",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "warmup_ratio",
        "logging_steps",
        "save_steps",
        "eval_steps",
        "eval_strategy",
        "seed",
        "fp16",
        "bf16",
    ]:
        value = getattr(cli_args, key, None)
        if value is not None:
            setattr(cfg, key, value)
    if cli_args.output_dir:
        cfg.output_dir = cli_args.output_dir
    return cfg


def _init_npu() -> None:
    try:
        import torch_npu  # type: ignore

        torch.npu.set_device(0)
        _ = torch_npu
    except Exception as exc:
        raise RuntimeError(
            "torch_npu is required for training. Run this script inside the Ascend NPU environment."
        ) from exc


def _make_dataset(examples: List[SupervisedExample]) -> Dataset:
    return Dataset.from_dict(
        {
            "text": [ex.text for ex in examples],
            "label": [ex.label_id for ex in examples],
        }
    )


def _tokenize_function(tokenizer, max_length: int):
    def _tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    return _tokenize


_ACCURACY_METRIC = evaluate.load("accuracy")
_F1_METRIC = evaluate.load("f1")


def _compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = _ACCURACY_METRIC.compute(predictions=preds, references=labels)["accuracy"]
    f1_macro = _F1_METRIC.compute(predictions=preds, references=labels, average="macro")["f1"]
    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
    }


def _evaluate_on_test(
    trainer: Trainer,
    test_dataset: Dataset,
    test_tokenized: Dataset,
    id2label: Dict[int, str],
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    raw_pred = trainer.predict(test_tokenized)
    logits = raw_pred.predictions
    labels = raw_pred.label_ids
    probs = torch.softmax(torch.tensor(logits), dim=-1).cpu().numpy()
    pred_ids = np.argmax(probs, axis=-1)

    rows: List[Dict[str, object]] = []
    for idx, (text, gold_id, pred_id, prob_row) in enumerate(
        zip(test_dataset["text"], labels, pred_ids, probs)
    ):
        prob_map = {id2label[i]: float(prob_row[i]) for i in range(len(id2label))}
        rows.append(
            {
                "idx": idx,
                "text": text,
                "gold_id": int(gold_id),
                "gold": id2label[int(gold_id)],
                "pred_id": int(pred_id),
                "pred": id2label[int(pred_id)],
                "pred_conf": float(prob_row[int(pred_id)]),
                "probs": prob_map,
            }
        )

    metric_values = _compute_metrics((logits, labels))
    return {
        "accuracy": float(metric_values["accuracy"]),
        "f1_macro": float(metric_values["f1_macro"]),
    }, rows


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _write_config_snapshot(path: Path, cfg: TrainConfig) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump({"train": cfg.__dict__}, handle, allow_unicode=True, sort_keys=False)


def _write_git_commit(path: Path) -> None:
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        commit = result.stdout.strip()
    except Exception:
        commit = "unknown"

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(commit + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to dataset JSON")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--config", default="configs/classifier/train.yaml")

    parser.add_argument("--model_name_or_path")
    parser.add_argument("--max_length", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--num_train_epochs", type=int)
    parser.add_argument("--per_device_train_batch_size", type=int)
    parser.add_argument("--per_device_eval_batch_size", type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int)
    parser.add_argument("--warmup_ratio", type=float)
    parser.add_argument("--logging_steps", type=int)
    parser.add_argument("--save_steps", type=int)
    parser.add_argument("--eval_steps", type=int)
    parser.add_argument("--eval_strategy")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")

    args = parser.parse_args()

    _init_npu()

    cfg = _merge_config(args)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_config_snapshot(output_dir / "config_snapshot.yaml", cfg)
    _write_git_commit(output_dir / "git_commit.txt")

    examples = load_supervised_rows(args.data)
    train_rows, val_rows, test_rows = make_splits(
        examples,
        seed=cfg.seed,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name_or_path,
        num_labels=len(LABEL2ID),
        label2id=LABEL2ID,
        id2label={str(k): v for k, v in ID2LABEL.items()},
    )

    dataset = DatasetDict(
        {
            "train": _make_dataset(train_rows),
            "validation": _make_dataset(val_rows),
            "test": _make_dataset(test_rows),
        }
    )

    tokenized = dataset.map(
        _tokenize_function(tokenizer, cfg.max_length),
        batched=True,
        remove_columns=["text"],
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    args_train = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        eval_strategy=cfg.eval_strategy,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        seed=cfg.seed,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
    )

    trainer = Trainer(
        model=model,
        args=args_train,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=_compute_metrics,
    )

    trainer.train()

    test_metrics, pred_rows = _evaluate_on_test(
        trainer,
        test_dataset=dataset["test"],
        test_tokenized=tokenized["test"],
        id2label=ID2LABEL,
    )

    _write_json(output_dir / "test_metrics.json", test_metrics)
    _write_json(output_dir / "test_pred_vs_gold.json", pred_rows)
    _write_json(
        output_dir / "label_mapping.json",
        {"label2id": LABEL2ID, "id2label": ID2LABEL},
    )

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
