#!/usr/bin/env python3
"""Verify that Week 1 recordings are numerous, balanced, and well formed."""

from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path

LABELS = ("circle", "left_right", "up_down")
EXPECTED_HEADER = ("timestamp_ms", "ax", "ay", "az", "gx", "gy", "gz")
MIN_RECORDINGS_PER_LABEL = 15
MIN_ROWS = 100
MAX_ROWS = 250


def inspect_recording(path: Path) -> list[str]:
    errors: list[str] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = tuple(next(reader, ()))
        if header != EXPECTED_HEADER:
            return [f"{path.name}: expected header {','.join(EXPECTED_HEADER)}"]
        rows = list(reader)

    if not MIN_ROWS <= len(rows) <= MAX_ROWS:
        errors.append(
            f"{path.name}: {len(rows)} rows; expected {MIN_ROWS}–{MAX_ROWS}"
        )
    previous_time = -1.0
    for line_number, row in enumerate(rows, start=2):
        if len(row) != len(EXPECTED_HEADER):
            errors.append(f"{path.name}:{line_number}: expected 7 columns")
            continue
        try:
            values = [float(value) for value in row]
        except ValueError:
            errors.append(f"{path.name}:{line_number}: contains non-numeric data")
            continue
        if not all(math.isfinite(value) for value in values):
            errors.append(f"{path.name}:{line_number}: contains a non-finite value")
        if values[0] <= previous_time:
            errors.append(f"{path.name}:{line_number}: timestamps do not increase")
        previous_time = values[0]
        if len(errors) >= 10:
            errors.append(f"{path.name}: stopped after 10 errors")
            break
    return errors


def main() -> int:
    data_dir = Path(__file__).resolve().parent / "data"
    files = sorted(data_dir.glob("*.csv"))
    counts: Counter[str] = Counter()
    errors: list[str] = []

    for path in files:
        matching_labels = [label for label in LABELS if path.name.startswith(f"{label}_")]
        if len(matching_labels) != 1:
            errors.append(f"{path.name}: filename must start with a known label and '_'")
            continue
        counts[matching_labels[0]] += 1
        errors.extend(inspect_recording(path))

    for label in LABELS:
        if counts[label] < MIN_RECORDINGS_PER_LABEL:
            errors.append(
                f"{label}: found {counts[label]} recordings; need {MIN_RECORDINGS_PER_LABEL}"
            )

    if errors:
        print("Week 1 validation FAILED:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"Week 1 validation PASSED: {len(files)} valid recordings")
    for label in LABELS:
        print(f"- {label}: {counts[label]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
