#!/usr/bin/env python3
"""Verify quantization, deployable artifacts, measured quality, and demo proof."""

from __future__ import annotations

import json
import re
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    float_model = root / "week2-model-training" / "models" / "gesture_model.keras"
    lite_model = root / "week2-model-training" / "models" / "gesture_model_int8.tflite"
    metrics_path = root / "week2-model-training" / "models" / "metrics.json"
    header_path = (
        root
        / "week3-deployment"
        / "arduino"
        / "gesture_infer"
        / "model_data.h"
    )
    demo_dir = root / "week3-deployment" / "demo"
    errors: list[str] = []

    model_bytes = b""
    if not lite_model.exists():
        errors.append("quantized .tflite model is missing; run quantize.py")
    else:
        model_bytes = lite_model.read_bytes()
        if len(model_bytes) < 1000 or model_bytes[4:8] != b"TFL3":
            errors.append("quantized model is not a substantive TFLite flatbuffer")

    if not float_model.exists():
        errors.append("Week 2 Keras model is missing")
    elif model_bytes and len(model_bytes) >= float_model.stat().st_size:
        errors.append("quantized model is not smaller than the Keras model file")

    if not header_path.exists():
        errors.append("Arduino model_data.h is missing; run quantize.py")
    elif model_bytes:
        header = header_path.read_text(encoding="utf-8")
        length_match = re.search(r"g_model_len = (\d+);", header)
        byte_count = len(re.findall(r"0x[0-9a-fA-F]{2}", header))
        if not length_match or int(length_match.group(1)) != len(model_bytes):
            errors.append("model_data.h length does not match the .tflite model")
        if byte_count != len(model_bytes):
            errors.append("model_data.h byte array is incomplete")
        for required_name in ("kNormalizationMean", "kNormalizationStd", "kGestureLabels"):
            if required_name not in header:
                errors.append(f"model_data.h is missing {required_name}")

    if not metrics_path.exists():
        errors.append("Week 2 metrics.json is missing")
    else:
        try:
            accuracy = float(
                json.loads(metrics_path.read_text(encoding="utf-8"))["test_accuracy"]
            )
            if accuracy < 0.70:
                errors.append(f"measured test accuracy is only {accuracy:.1%}")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"could not read test accuracy: {error}")

    demo_files = (
        [
            path
            for path in demo_dir.iterdir()
            if path.suffix.lower() in {".gif", ".mp4", ".mov", ".webm"}
            and path.stat().st_size >= 1000
        ]
        if demo_dir.is_dir()
        else []
    )
    if not demo_files:
        errors.append("demo/ needs a substantive GIF or video of live inference")

    if errors:
        print("Week 3 validation FAILED:")
        for error in errors:
            print(f"- {error}")
        return 1

    reduction = 1 - len(model_bytes) / float_model.stat().st_size
    print(f"Week 3 validation PASSED: model is {len(model_bytes) / 1024:.1f} KiB")
    print(f"- file-size reduction from Keras model: {reduction:.1%}")
    print(f"- measured test accuracy: {accuracy:.1%}")
    print(f"- demo: {demo_files[0].name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
