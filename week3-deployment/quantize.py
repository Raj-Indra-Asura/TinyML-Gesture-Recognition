#!/usr/bin/env python3
"""Create a fully int8 TensorFlow Lite model and an Arduino C header."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=root / "week2-model-training" / "models" / "gesture_model.keras",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=root / "week2-model-training" / "prepared_data.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "week2-model-training" / "models" / "gesture_model_int8.tflite",
    )
    parser.add_argument(
        "--header",
        type=Path,
        default=Path(__file__).resolve().parent
        / "arduino"
        / "gesture_infer"
        / "model_data.h",
    )
    return parser.parse_args()


def format_float_array(name: str, values: np.ndarray) -> str:
    body = ", ".join(f"{float(value):.9g}f" for value in values)
    return f"constexpr float {name}[] = {{{body}}};"


def write_header(
    path: Path,
    model_bytes: bytes,
    mean: np.ndarray,
    std: np.ndarray,
    labels: list[str],
) -> None:
    byte_lines = []
    for offset in range(0, len(model_bytes), 12):
        chunk = model_bytes[offset : offset + 12]
        byte_lines.append("  " + ", ".join(f"0x{value:02x}" for value in chunk) + ",")
    label_values = ", ".join(f'"{label}"' for label in labels)
    contents = "\n".join(
        [
            "#ifndef GESTURE_MODEL_DATA_H",
            "#define GESTURE_MODEL_DATA_H",
            "",
            "#include <Arduino.h>",
            "",
            "alignas(16) const unsigned char g_model[] = {",
            *byte_lines,
            "};",
            f"constexpr unsigned int g_model_len = {len(model_bytes)};",
            format_float_array("kNormalizationMean", mean),
            format_float_array("kNormalizationStd", std),
            f"constexpr const char* kGestureLabels[] = {{{label_values}}};",
            f"constexpr int kGestureCount = {len(labels)};",
            "",
            "#endif",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def main() -> int:
    args = parse_args()
    try:
        import tensorflow as tf
    except ImportError:
        print("Missing TensorFlow. Run: python -m pip install tensorflow")
        return 2
    if not args.model.exists() or not args.data.exists():
        print("Missing Week 2 model or prepared data. Complete Week 2 first.")
        return 1

    try:
        model = tf.keras.models.load_model(args.model)
        with np.load(args.data) as data:
            representative_values = data["x_train"].astype(np.float32)
            mean = data["normalization_mean"].astype(np.float32)
            std = data["normalization_std"].astype(np.float32)
            labels = [str(value) for value in data["labels"]]
    except (KeyError, OSError, ValueError) as error:
        print(f"Could not load Week 2 artifacts: {error}")
        return 1

    def representative_dataset():
        for sample in representative_values:
            yield [sample[np.newaxis, ...]]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    try:
        model_bytes = converter.convert()
    except (RuntimeError, ValueError) as error:
        print(f"TensorFlow Lite conversion failed: {error}")
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(model_bytes)
    write_header(args.header, model_bytes, mean, std, labels)

    interpreter = tf.lite.Interpreter(model_content=model_bytes)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    if input_detail["dtype"] != np.int8 or output_detail["dtype"] != np.int8:
        print("Conversion did not create int8 input and output tensors.")
        return 1

    float_size = args.model.stat().st_size
    print(f"Saved {args.output} ({len(model_bytes) / 1024:.1f} KiB)")
    print(f"Float Keras file: {float_size / 1024:.1f} KiB")
    print(f"Saved Arduino header to {args.header}")
    print(f"Input quantization (scale, zero point): {input_detail['quantization']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
