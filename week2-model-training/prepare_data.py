#!/usr/bin/env python3
"""Convert Week 1 recordings into leak-free train, validation, and test arrays."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

LABELS = ("circle", "left_right", "up_down")
FEATURES = ("ax", "ay", "az", "gx", "gy", "gz")
WINDOW_SIZE = 128
SEED = 42


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=root / "week1-data-foundations" / "data"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "prepared_data.npz",
    )
    return parser.parse_args()


def label_for(path: Path) -> str:
    matches = [label for label in LABELS if path.name.startswith(f"{label}_")]
    if len(matches) != 1:
        raise ValueError(f"{path.name}: filename does not begin with one known label")
    return matches[0]


def read_recording(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["timestamp_ms", *FEATURES]:
            raise ValueError(f"{path.name}: unexpected CSV header")
        try:
            values = np.asarray(
                [[float(row[feature]) for feature in FEATURES] for row in reader],
                dtype=np.float32,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path.name}: invalid numeric data") from error
    if len(values) < 10 or not np.isfinite(values).all():
        raise ValueError(f"{path.name}: too few rows or non-finite data")
    return values


def resample(values: np.ndarray) -> np.ndarray:
    old_positions = np.linspace(0.0, 1.0, num=len(values))
    new_positions = np.linspace(0.0, 1.0, num=WINDOW_SIZE)
    columns = [
        np.interp(new_positions, old_positions, values[:, index])
        for index in range(values.shape[1])
    ]
    return np.stack(columns, axis=1).astype(np.float32)


def split_indices(labels: np.ndarray) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED)
    splits: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    for label_index in range(len(LABELS)):
        indices = np.flatnonzero(labels == label_index)
        if len(indices) < 10:
            raise ValueError(f"{LABELS[label_index]} needs at least 10 recordings")
        rng.shuffle(indices)
        train_end = int(len(indices) * 0.70)
        validation_end = train_end + max(1, int(len(indices) * 0.15))
        splits["train"].extend(indices[:train_end])
        splits["validation"].extend(indices[train_end:validation_end])
        splits["test"].extend(indices[validation_end:])
    return {
        name: rng.permutation(np.asarray(indices, dtype=np.int64))
        for name, indices in splits.items()
    }


def main() -> int:
    args = parse_args()
    paths = sorted(args.data_dir.glob("*.csv"))
    if not paths:
        print(f"No CSV recordings found in {args.data_dir}. Complete Week 1 first.")
        return 1

    try:
        raw_windows = np.stack([resample(read_recording(path)) for path in paths])
        labels = np.asarray([LABELS.index(label_for(path)) for path in paths])
        splits = split_indices(labels)
    except ValueError as error:
        print(f"Could not prepare data: {error}")
        return 1

    train_values = raw_windows[splits["train"]]
    mean = train_values.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = train_values.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = np.maximum(std, np.float32(1e-6))
    normalized = (raw_windows - mean) / std

    arrays: dict[str, np.ndarray] = {
        "labels": np.asarray(LABELS),
        "feature_names": np.asarray(FEATURES),
        "normalization_mean": mean,
        "normalization_std": std,
    }
    for name, indices in splits.items():
        arrays[f"x_{name}"] = normalized[indices].astype(np.float32)
        arrays[f"y_{name}"] = labels[indices].astype(np.int64)
        arrays[f"recording_{name}"] = np.asarray([paths[i].name for i in indices])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(f"Saved {len(paths)} recordings to {args.output}")
    for name in splits:
        counts = defaultdict(int)
        for value in arrays[f"y_{name}"]:
            counts[LABELS[int(value)]] += 1
        print(f"- {name}: {len(splits[name])} {dict(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
