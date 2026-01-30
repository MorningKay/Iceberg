"""Dataset loading and splitting for the Iceberg classifier."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


LABEL2ID: Dict[str, int] = {
    "behavior": 0,
    "coping": 1,
    "feelings": 2,
    "feelings about feelings": 3,
    "perceptions": 4,
}
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}


@dataclass(frozen=True)
class SupervisedExample:
    text: str
    label_name: str
    label_id: int


def _iter_speaker_rows(samples: Iterable[dict]) -> Iterable[SupervisedExample]:
    for sample in samples:
        dialogue = sample.get("dialogue", [])
        for round_item in dialogue:
            turns = round_item.get("turns", [])
            for turn in turns:
                if turn.get("role") != "Speaker":
                    continue
                text = (turn.get("text") or "").strip()
                if not text:
                    continue
                label_name = turn.get("iceberg_layer")
                if label_name not in LABEL2ID:
                    continue
                yield SupervisedExample(
                    text=text,
                    label_name=label_name,
                    label_id=LABEL2ID[label_name],
                )


def load_supervised_rows(json_path: str) -> List[SupervisedExample]:
    """Load supervised rows from a JSON file.

    The JSON top-level must be a list of samples.
    """

    with open(json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Dataset JSON must be a top-level list")
    return list(_iter_speaker_rows(data))


def _shuffle_indices(count: int, seed: int) -> List[int]:
    rng = __import__("random")
    random = rng.Random(seed)
    indices = list(range(count))
    random.shuffle(indices)
    return indices


def split_indices(
    labels: Sequence[int],
    seed: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
) -> Tuple[List[int], List[int], List[int]]:
    if not labels:
        raise ValueError("No labels provided for splitting")
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    try:
        from sklearn.model_selection import train_test_split
    except Exception:
        train_test_split = None

    indices = list(range(len(labels)))
    if train_test_split is None:
        shuffled = _shuffle_indices(len(labels), seed)
        train_end = int(len(labels) * train_ratio)
        val_end = train_end + int(len(labels) * val_ratio)
        train_idx = shuffled[:train_end]
        val_idx = shuffled[train_end:val_end]
        test_idx = shuffled[val_end:]
        return train_idx, val_idx, test_idx

    try:
        train_idx, temp_idx = train_test_split(
            indices,
            test_size=(1.0 - train_ratio),
            random_state=seed,
            shuffle=True,
            stratify=list(labels),
        )
        temp_labels = [labels[i] for i in temp_idx]
        val_fraction = val_ratio / (val_ratio + test_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=(1.0 - val_fraction),
            random_state=seed,
            shuffle=True,
            stratify=temp_labels,
        )
        return list(train_idx), list(val_idx), list(test_idx)
    except ValueError:
        shuffled = _shuffle_indices(len(labels), seed)
        train_end = int(len(labels) * train_ratio)
        val_end = train_end + int(len(labels) * val_ratio)
        train_idx = shuffled[:train_end]
        val_idx = shuffled[train_end:val_end]
        test_idx = shuffled[val_end:]
        return train_idx, val_idx, test_idx


def make_splits(
    examples: Sequence[SupervisedExample],
    seed: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
) -> Tuple[List[SupervisedExample], List[SupervisedExample], List[SupervisedExample]]:
    labels = [ex.label_id for ex in examples]
    train_idx, val_idx, test_idx = split_indices(
        labels,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    train_set = [examples[i] for i in train_idx]
    val_set = [examples[i] for i in val_idx]
    test_set = [examples[i] for i in test_idx]
    return train_set, val_set, test_set
