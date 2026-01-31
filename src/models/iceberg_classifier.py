"""Iceberg Layer classifier inference wrapper.

Usage:
  clf = IcebergClassifier("/path/to/exported/checkpoint")
  label = clf.predict_label("some text")
  label, probs = clf.predict_with_probs("some text")

Expected model_dir layout includes model weights/config and tokenizer files
(e.g., config.json, model.safetensors, tokenizer.json). If a
label_mapping.json is present, it must match the fixed Iceberg mapping.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = [
    "behavior",
    "coping",
    "feelings",
    "feelings about feelings",
    "perceptions",
]
LABEL2ID: Dict[str, int] = {
    "behavior": 0,
    "coping": 1,
    "feelings": 2,
    "feelings about feelings": 3,
    "perceptions": 4,
}
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}


def _load_label_mapping(model_dir: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    mapping_path = model_dir / "label_mapping.json"
    if not mapping_path.exists():
        return dict(LABEL2ID), dict(ID2LABEL)

    with open(mapping_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    label2id = payload.get("label2id")
    id2label = payload.get("id2label")
    if label2id != LABEL2ID:
        raise ValueError("label_mapping.json label2id does not match fixed mapping")
    expected_id2label = {str(k): v for k, v in ID2LABEL.items()}
    if id2label != expected_id2label:
        raise ValueError("label_mapping.json id2label does not match fixed mapping")

    return dict(LABEL2ID), dict(ID2LABEL)


def _resolve_device(device: str) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if device not in {"auto", "npu"}:
        raise ValueError("device must be one of: auto, cpu, npu")

    try:
        import torch_npu  # type: ignore

        _ = torch_npu
    except Exception as exc:
        raise RuntimeError(
            "torch_npu is required for NPU inference. Use device='cpu' or run in the Ascend NPU environment."
        ) from exc

    if not hasattr(torch, "npu"):
        raise RuntimeError("torch.npu is not available in this environment")

    torch.npu.set_device(0)
    return torch.device("npu:0")


class IcebergClassifier:
    def __init__(self, model_dir: str, device: str = "auto", local_files_only: bool = True):
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"model_dir not found: {model_dir}")

        self.label2id, self.id2label = _load_label_mapping(self.model_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir),
            local_files_only=local_files_only,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_dir),
            local_files_only=local_files_only,
        )

        self.device = _resolve_device(device)
        self.model.to(self.device)
        self.model.eval()

    def predict_label(self, text: str) -> str:
        label, _ = self.predict_with_probs(text)
        return label

    def predict_with_probs(self, text: str) -> Tuple[str, Dict[str, float]]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

        if not torch.isfinite(probs).all():
            raise RuntimeError("Non-finite probabilities produced by the model")

        probs = probs.squeeze(0).detach().cpu().numpy().tolist()
        pred_id = int(max(range(len(probs)), key=lambda idx: probs[idx]))
        pred_label = self.id2label[pred_id]

        prob_map = {self.id2label[i]: float(probs[i]) for i in range(len(self.id2label))}
        return pred_label, prob_map
