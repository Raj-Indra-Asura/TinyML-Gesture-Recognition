#!/usr/bin/env python3
"""Collect one labeled IMU recording from the Arduino serial stream."""

from __future__ import annotations

import argparse
import csv
import math
import time
from datetime import datetime, timezone
from pathlib import Path

LABELS = ("circle", "left_right", "up_down")
HEADER = ("timestamp_ms", "ax", "ay", "az", "gx", "gy", "gz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="Serial port, such as COM4 or /dev/ttyACM0")
    parser.add_argument("--label", required=True, choices=LABELS)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
    )
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be greater than zero")
    return args


def parse_sensor_row(line: str) -> tuple[float, ...] | None:
    parts = line.strip().split(",")
    if len(parts) != 6:
        return None
    try:
        values = tuple(float(part) for part in parts)
    except ValueError:
        return None
    return values if all(math.isfinite(value) for value in values) else None


def main() -> int:
    try:
        import serial
    except ImportError:
        print("Missing pyserial. Run: python -m pip install pyserial")
        return 2

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = args.output_dir / f"{args.label}_{stamp}.csv"

    rows: list[tuple[float, ...]] = []
    malformed = 0
    print(f"Opening {args.port}. Keep the board still while the connection starts...")
    try:
        with serial.Serial(args.port, args.baud, timeout=1) as board:
            time.sleep(2)
            board.reset_input_buffer()
            print(f"GO: perform {args.label!r} smoothly for {args.seconds:g} seconds.")
            started = time.monotonic()
            while time.monotonic() - started < args.seconds:
                line = board.readline().decode("utf-8", errors="replace")
                values = parse_sensor_row(line)
                if values is None:
                    malformed += 1
                    continue
                elapsed_ms = (time.monotonic() - started) * 1000
                rows.append((elapsed_ms, *values))
    except OSError as error:
        print(f"Could not read {args.port}: {error}")
        return 2

    if len(rows) < 10:
        print(f"Only received {len(rows)} valid rows; no file was saved.")
        return 1

    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {destination}")
    if malformed:
        print(f"Ignored {malformed} startup or malformed lines.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
