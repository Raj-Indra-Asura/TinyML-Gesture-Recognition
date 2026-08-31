#!/usr/bin/env python3
"""Run live gesture inference on the laptop from Arduino IMU serial data."""

from __future__ import annotations

import argparse
import math
import time
from collections import deque
from pathlib import Path

import numpy as np

WINDOW_SIZE = 128


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="Serial port, such as COM4 or /dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument(
        "--model",
        type=Path,
        default=root / "week2-model-training" / "models" / "gesture_model_int8.tflite",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=root / "week2-model-training" / "prepared_data.npz",
    )
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be from 0 to 1")
    return args


def parse_sensor_row(line: str) -> np.ndarray | None:
    parts = line.strip().split(",")
    if len(parts) != 6:
        return None
    try:
        values = np.asarray([float(part) for part in parts], dtype=np.float32)
    except ValueError:
        return None
    return values if np.isfinite(values).all() else None


def load_interpreter(model_path: Path):
    try:
        from tensorflow.lite import Interpreter
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError as error:
            raise RuntimeError(
                "Install TensorFlow or tflite-runtime before live inference"
            ) from error
    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter


def quantize(values: np.ndarray, detail: dict) -> np.ndarray:
    dtype = detail["dtype"]
    if np.issubdtype(dtype, np.floating):
        return values.astype(dtype)
    scale, zero_point = detail["quantization"]
    if scale <= 0:
        raise RuntimeError("Model input has invalid quantization metadata")
    limits = np.iinfo(dtype)
    transformed = np.rint(values / scale + zero_point)
    return np.clip(transformed, limits.min, limits.max).astype(dtype)


def dequantize(values: np.ndarray, detail: dict) -> np.ndarray:
    if np.issubdtype(detail["dtype"], np.floating):
        return values.astype(np.float32)
    scale, zero_point = detail["quantization"]
    return (values.astype(np.float32) - zero_point) * scale


def main() -> int:
    args = parse_args()
    try:
        import serial
    except ImportError:
        print("Missing pyserial. Run: python -m pip install pyserial")
        return 2
    if not args.model.exists() or not args.data.exists():
        print("Missing quantized model or prepared data. Run quantize.py first.")
        return 1

    try:
        interpreter = load_interpreter(args.model)
        with np.load(args.data) as data:
            mean = data["normalization_mean"].astype(np.float32)
            std = data["normalization_std"].astype(np.float32)
            labels = [str(value) for value in data["labels"]]
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"Could not load inference artifacts: {error}")
        return 1

    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    window: deque[np.ndarray] = deque(maxlen=WINDOW_SIZE)
    print("Hold still while serial starts. Press Ctrl+C to stop.")
    try:
        with serial.Serial(args.port, args.baud, timeout=1) as board:
            time.sleep(2)
            board.reset_input_buffer()
            while True:
                row = parse_sensor_row(
                    board.readline().decode("utf-8", errors="replace")
                )
                if row is None:
                    continue
                window.append(row)
                if len(window) < WINDOW_SIZE:
                    continue
                normalized = (np.stack(window) - mean) / std
                input_value = quantize(normalized[np.newaxis, ...], input_detail)
                interpreter.set_tensor(input_detail["index"], input_value)
                interpreter.invoke()
                probabilities = dequantize(
                    interpreter.get_tensor(output_detail["index"])[0], output_detail
                )
                best = int(np.argmax(probabilities))
                confidence = float(probabilities[best])
                prediction = labels[best] if confidence >= args.threshold else "uncertain"
                print(f"{prediction:>12}  confidence={confidence:.1%}")
                window.clear()
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0
    except OSError as error:
        print(f"Could not read {args.port}: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
