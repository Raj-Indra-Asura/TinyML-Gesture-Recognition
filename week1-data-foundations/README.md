# Week 1 — Data Foundations (project 0% → 35%)

This folder is the first of three build stages. Read this page first for the
overview, then work through [`learning.md`](learning.md), which contains every
instruction, explanation, and definition you need. You do not need any other
book, video, or website to finish this week.

## Overview

The project is a gesture recognizer: you wave an Arduino board in the air and
the computer names the movement. A recognizer can only be as good as the
examples it learns from, so Week 1 builds the dataset.

By the end of this week you will have:

1. An Arduino Nano 33 BLE Sense Rev2 that streams six motion numbers
   (`ax, ay, az, gx, gy, gz`) to your laptop 50 times per second.
2. A Python collector that saves one labeled three-second movement per file.
3. At least 45 recordings — 15 each of `circle`, `left_right`, and `up_down`.
4. A validator that proves the files are complete, balanced, and well formed.

Nothing is trained yet. Week 1 produces the raw material that Weeks 2 and 3
turn into a working model.

## What this week contributes to the whole project

| Stage | Delivered this week | Cumulative project completion |
| --- | --- | --- |
| Week 1 — Data Foundations | Hardware brought up, sensor streaming, labeled dataset collected and validated | **35%** |
| Week 2 — Model Training | Data preparation, trained model, measured accuracy | 75% |
| Week 3 — Deployment | Quantization, live inference, demo and honest limitations | 100% |

The 35% is the hardware bring-up plus the dataset. It is the largest single
risk in the project: a model cannot repair unclear or wrongly labeled
recordings, so time spent here saves time in Weeks 2 and 3.

## Before you start

- Arduino Nano 33 BLE Sense Rev2 and a **data-capable** USB cable
- Arduino IDE 2 installed, with the `Arduino_BMI270_BMM150` library
- Python 3.10 or newer with `pyserial` installed
- About an arm's length of clear desk space for moving the board

Exact installation steps, including how to tell a charge-only cable from a data
cable, are in [`learning.md`](learning.md).

## Files in this folder

| Path | Purpose |
| --- | --- |
| `README.md` | This overview |
| `learning.md` | The complete Week 1 lesson: concepts, daily instructions, code walkthrough, troubleshooting |
| `arduino/stream_imu/stream_imu.ino` | Sketch that reads the sensor and prints one CSV row per sample |
| `collect_data.py` | Records one labeled gesture into `data/` |
| `validate_week1.py` | Checks that the dataset is complete and well formed |
| `data/` | Your recordings (ignored by Git; `.gitkeep` keeps the folder) |

## Daily map

| Day | Focus | Evidence you are done |
| --- | --- | --- |
| 1 | Board, IDE, and sensor stream | Six changing numbers in Serial Monitor |
| 2 | First practice recording | A CSV with 7 columns and roughly 150 rows |
| 3 | Physical collection | 15 recordings per gesture |
| 4 | Quality inspection | No known mislabeled or interrupted take |
| 5 | Proof | `validate_week1.py` prints `PASSED` |

## How Week 1 is checked

Run this from the repository root:

```bash
python week1-data-foundations/validate_week1.py
```

It passes when every CSV has the header
`timestamp_ms,ax,ay,az,gx,gy,gz`, has 100–250 rows, has increasing timestamps,
holds only finite numbers, and each of the three labels has at least 15 files.

## Repository structure at the end of Week 1

Items marked *(generated, not committed)* are created on your machine by the
commands in this week and are excluded by `.gitignore`.

```text
TinyML-Gesture-Recognition/
├── .gitignore
├── NOTES.md                              # your experiment log
├── PROGRESS.md                           # course checklist
├── README.md                             # course entry point
├── week1-data-foundations/               # ACTIVE THIS WEEK
│   ├── README.md                         # this overview
│   ├── learning.md                       # the complete Week 1 lesson
│   ├── collect_data.py
│   ├── validate_week1.py
│   ├── arduino/
│   │   └── stream_imu/
│   │       └── stream_imu.ino
│   └── data/
│       ├── .gitkeep
│       ├── circle_20240101T101500Z.csv        # (generated, not committed)
│       ├── ... 15 or more circle recordings
│       ├── left_right_20240101T101800Z.csv    # (generated, not committed)
│       ├── ... 15 or more left_right recordings
│       ├── up_down_20240101T102100Z.csv       # (generated, not committed)
│       └── ... 15 or more up_down recordings
├── week2-model-training/                 # not used yet
│   ├── README.md
│   ├── learning.md
│   ├── prepare_data.py
│   ├── train.ipynb
│   ├── validate_week2.py
│   └── models/
│       └── .gitkeep
└── week3-deployment/                     # not used yet
    ├── README.md
    ├── learning.md
    ├── quantize.py
    ├── live_infer.py
    ├── validate_week3.py
    ├── arduino/
    │   └── gesture_infer/
    │       └── gesture_infer.ino
    └── demo/
        └── README.md
```

Only `data/.gitkeep` and the tracked source files above are in Git at this
point. Your recordings stay on your machine because they describe your
movements and can always be recollected.

## Next step

Open [`learning.md`](learning.md) and start Day 1. When
`validate_week1.py` prints `PASSED`, tick the Week 1 boxes in
[`../PROGRESS.md`](../PROGRESS.md) and continue to
[Week 2](../week2-model-training/README.md).
