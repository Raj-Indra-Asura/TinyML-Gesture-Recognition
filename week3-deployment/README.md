# Week 3 — Deployment (project 75% → 100%)

This folder is the final build stage. Read this page first for the overview,
then work through [`learning.md`](learning.md), which contains every
instruction, explanation, and definition you need. You do not need any other
book, video, or website to finish this week.

## Overview

Week 2 produced a model that scores well on saved recordings. Week 3 makes it
a working product: smaller, running on live sensor data, measured in the real
world, and described honestly.

By the end of this week you will have:

1. `gesture_model_int8.tflite` — the same model with 8-bit integer weights,
   several times smaller than the Keras file.
2. `model_data.h` — that model as a C array so an Arduino sketch with no
   filesystem can hold it in flash memory.
3. Live inference running either on the Arduino
   (`arduino/gesture_infer/gesture_infer.ino`) or on your laptop
   (`live_infer.py`), both executing the same quantized model.
4. A short demo recording in `demo/` plus a written statement of accuracy,
   model size, and known failure cases.

## What this week contributes to the whole project

| Stage | Delivered | Cumulative project completion |
| --- | --- | --- |
| Week 1 — Data Foundations | Hardware bring-up and validated dataset | 35% |
| Week 2 — Model Training | Trained model and measured accuracy | 75% |
| Week 3 — Deployment | Quantization, C header, live inference, field test, demo, limitations | **100%** |

Week 3 adds the final 25%. When it is finished the project is complete: raw
motion goes in one end and a named gesture comes out the other, with evidence
for every claim.

## Before you start

- `python week2-model-training/validate_week2.py` must print `PASSED`.
  Week 3 converts Week 2's exact saved model; do not begin without it.
- TensorFlow installed (used by `quantize.py`, and by `live_infer.py` unless
  you install the smaller `tflite-runtime`).
- For the on-device path only: the `Chirale_TensorFlowLite` Arduino library
  alongside `Arduino_BMI270_BMM150`.

## Two deployment paths

Both paths run the identical `int8` model on live sensor data. Choose one and
say which one you used when you present the project.

| | Path A — Arduino | Path B — laptop fallback |
| --- | --- | --- |
| Where inference runs | On the microcontroller | On your laptop |
| Sensor data | Live | Live, streamed over USB |
| Extra setup | `Chirale_TensorFlowLite` library | None beyond Week 2 |
| Use it when | The library compiles for your board | Library or toolchain problems block you |

Path B is a genuine, honest result. Do not spend the week fighting a library
installation; record the failure in `NOTES.md` and use Path B.

## Files in this folder

| Path | Purpose |
| --- | --- |
| `README.md` | This overview |
| `learning.md` | The complete Week 3 lesson: concepts, daily instructions, code walkthrough, troubleshooting |
| `quantize.py` | Converts the Keras model to `int8` TFLite and writes `model_data.h` |
| `live_infer.py` | Laptop inference on live serial data (Path B) |
| `arduino/gesture_infer/gesture_infer.ino` | On-device inference sketch (Path A) |
| `validate_week3.py` | Re-checks the converted model, the header, the retained accuracy, and the demo |
| `demo/` | Your demo GIF or video (ignored by Git; `README.md` explains what to capture) |

## Daily map

| Day | Focus | Evidence you are done |
| --- | --- | --- |
| 1 | Quantize | `.tflite` and `model_data.h` exist; size reduction recorded |
| 2 | Choose one live path | One 128-sample window produces a label and confidence |
| 3 | Field test | Ten trials per gesture plus stillness, tallied honestly |
| 4 | Limitations and demo | Demo captured; accuracy, size, and limits written down |
| 5 | Proof | `validate_week3.py` prints `PASSED` |

## How Week 3 is checked

```bash
python week3-deployment/validate_week3.py
```

It passes when the `.tflite` file is a real TFLite flatbuffer smaller than the
Keras model, `model_data.h` contains exactly as many bytes as the model plus
the normalization values and label list, Week 2's measured accuracy is still at
least 70%, and `demo/` holds a substantive GIF or video.

## Repository structure at the end of Week 3 (project complete)

Items marked *(generated, not committed)* are created on your machine and are
excluded by `.gitignore`.

```text
TinyML-Gesture-Recognition/
├── .gitignore
├── NOTES.md                              # final accuracy, model size, failure cases
├── PROGRESS.md                           # every box ticked
├── README.md
├── week1-data-foundations/               # COMPLETE
│   ├── README.md
│   ├── learning.md
│   ├── collect_data.py
│   ├── validate_week1.py
│   ├── arduino/
│   │   └── stream_imu/
│   │       └── stream_imu.ino
│   └── data/
│       ├── .gitkeep
│       ├── circle_*.csv                       # (generated, not committed) 15+
│       ├── left_right_*.csv                   # (generated, not committed) 15+
│       └── up_down_*.csv                      # (generated, not committed) 15+
├── week2-model-training/                 # COMPLETE
│   ├── README.md
│   ├── learning.md
│   ├── prepare_data.py
│   ├── train.ipynb
│   ├── validate_week2.py
│   ├── prepared_data.npz                      # (generated, not committed)
│   └── models/
│       ├── .gitkeep
│       ├── gesture_model.keras                # (generated, not committed)
│       ├── gesture_model_int8.tflite          # (generated, not committed) NEW
│       ├── metrics.json                       # (generated, not committed)
│       └── confusion_matrix.png               # (generated, not committed)
└── week3-deployment/                     # ACTIVE THIS WEEK
    ├── README.md                         # this overview
    ├── learning.md                       # the complete Week 3 lesson
    ├── quantize.py
    ├── live_infer.py
    ├── validate_week3.py
    ├── arduino/
    │   └── gesture_infer/
    │       ├── gesture_infer.ino
    │       └── model_data.h                   # (generated, not committed) NEW
    └── demo/
        ├── README.md
        └── gesture_demo.gif                   # (generated, not committed) NEW
```

Note that `quantize.py` writes the `.tflite` file into
`week2-model-training/models/` next to the Keras model it came from, and writes
`model_data.h` next to the sketch that includes it.

## Finishing the project

All three commands must pass from the repository root:

```bash
python week1-data-foundations/validate_week1.py
python week2-model-training/validate_week2.py
python week3-deployment/validate_week3.py
```

Then complete [`../PROGRESS.md`](../PROGRESS.md) and record the final numbers
and limitations in [`../NOTES.md`](../NOTES.md). At that point the project is
100% done.
