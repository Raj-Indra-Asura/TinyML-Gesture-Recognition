#!/usr/bin/env python3
"""Verify real Week 2 arrays, model output, and measured test results."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

LABELS = ("circle", "left_right", "up_down")
MIN_TEST_ACCURACY = 0.70


def main() -> int:
    folder = Path(__file__).resolve().parent
    prepared_path = folder / "prepared_data.npz"
    model_path = folder / "models" / "gesture_model.keras"
    metrics_path = folder / "models" / "metrics.json"
    confusion_path = folder / "models" / "confusion_matrix.png"
    errors: list[str] = []

    if not prepared_path.exists():
        errors.append("prepared_data.npz is missing; run prepare_data.py")
    else:
        try:
            with np.load(prepared_path) as data:
                names: set[str] = set()
                for split in ("train", "validation", "test"):
                    x = data[f"x_{split}"]
                    y = data[f"y_{split}"]
                    recordings = data[f"recording_{split}"]
                    if x.ndim != 3 or x.shape[1:] != (128, 6):
                        errors.append(f"x_{split} has shape {x.shape}, expected (*, 128, 6)")
                    if len(x) != len(y) or len(y) != len(recordings):
                        errors.append(f"{split} arrays have different lengths")
                    overlap = names.intersection(str(item) for item in recordings)
                    if overlap:
                        errors.append(f"recordings leak between splits: {sorted(overlap)}")
                    names.update(str(item) for item in recordings)
                if not np.isfinite(data["x_train"]).all():
                    errors.append("training data contains non-finite values")
        except (KeyError, OSError, ValueError) as error:
            errors.append(f"could not inspect prepared_data.npz: {error}")

    if not model_path.exists() or model_path.stat().st_size < 1000:
        errors.append("models/gesture_model.keras is missing or too small")
    if not confusion_path.exists() or confusion_path.stat().st_size < 1000:
        errors.append("models/confusion_matrix.png is missing or too small")

    if not metrics_path.exists():
        errors.append("models/metrics.json is missing")
    else:
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            accuracy = float(metrics["test_accuracy"])
            matrix = metrics["confusion_matrix"]
            per_class = metrics["per_class_recall"]
            if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
                errors.append("test_accuracy must be a finite number from 0 to 1")
            elif accuracy < MIN_TEST_ACCURACY:
                errors.append(
                    f"test_accuracy is {accuracy:.3f}; need at least {MIN_TEST_ACCURACY:.2f}"
                )
            if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
                errors.append("confusion_matrix must be a 3 by 3 array")
            if set(per_class) != set(LABELS):
                errors.append("per_class_recall must contain every gesture label")
            if sum(sum(int(value) for value in row) for row in matrix) <= 0:
                errors.append("confusion_matrix does not contain evaluated examples")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"metrics.json is invalid: {error}")

    if errors:
        print("Week 2 validation FAILED:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"Week 2 validation PASSED: test accuracy {accuracy:.1%}")
    print(f"- evaluated recordings: {sum(sum(row) for row in matrix)}")
    print(f"- model size: {model_path.stat().st_size / 1024:.1f} KiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
