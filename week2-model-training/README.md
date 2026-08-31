# Week 2 — Model Training (project 35% → 75%)

This folder is the second of three build stages. Read this page first for the
overview, then work through [`learning.md`](learning.md), which contains every
instruction, explanation, and definition you need. You do not need any other
book, video, or website to finish this week.

## Overview

Week 1 gave you 45 or more labeled recordings of hand movements. Week 2 turns
those recordings into a trained neural network and, just as importantly, into
an honest measurement of how good that network is.

By the end of this week you will have:

1. `prepared_data.npz` — every recording resampled to a fixed 128 × 6 window,
   normalized, and split into training, validation, and test groups with no
   recording appearing in two groups.
2. A trained Keras model saved as `models/gesture_model.keras`.
3. `models/metrics.json` and `models/confusion_matrix.png` — the accuracy,
   per-gesture recall, and confusion matrix measured on recordings the model
   never trained on.
4. At least one controlled experiment written into `NOTES.md`.

The model runs on your laptop only. Making it small enough for the
microcontroller is Week 3's job.

## What this week contributes to the whole project

| Stage | Delivered | Cumulative project completion |
| --- | --- | --- |
| Week 1 — Data Foundations | Hardware bring-up and validated dataset | 35% |
| Week 2 — Model Training | Windowing, normalization, leak-free splits, trained model, measured accuracy | **75%** |
| Week 3 — Deployment | Quantization, live inference, demo and honest limitations | 100% |

Week 2 adds 40%: the entire learning pipeline and the evidence that it works.
After this week the project can recognize gestures — it just cannot do it yet
on the board, in real time.

## Before you start

- `python week1-data-foundations/validate_week1.py` must print `PASSED`.
  Week 2 reads Week 1's CSV files directly; do not begin with an incomplete
  dataset.
- The same Python environment as Week 1, plus `numpy`, `matplotlib`,
  `tensorflow`, and `jupyter`.

## Files in this folder

| Path | Purpose |
| --- | --- |
| `README.md` | This overview |
| `learning.md` | The complete Week 2 lesson: concepts, daily instructions, code walkthrough, troubleshooting |
| `prepare_data.py` | Reads Week 1 CSVs and writes `prepared_data.npz` |
| `train.ipynb` | Jupyter notebook that builds, trains, evaluates, and saves the model |
| `validate_week2.py` | Independently re-checks the arrays, the model file, and the metrics |
| `models/` | Trained artifacts (ignored by Git; `.gitkeep` keeps the folder) |
| `prepared_data.npz` | Prepared arrays (generated; ignored by Git) |

## Daily map

| Day | Focus | Evidence you are done |
| --- | --- | --- |
| 1 | Prepare examples | Train/validation/test counts printed, all three classes in each split |
| 2 | Train | Loss and accuracy curves visible, early stopping restored the best weights |
| 3 | Evaluate | Test accuracy, per-class recall, and confusion matrix recorded |
| 4 | One controlled experiment | 48-unit versus 24-unit comparison in `NOTES.md` |
| 5 | Proof | `validate_week2.py` prints `PASSED` |

## How Week 2 is checked

```bash
python week2-model-training/validate_week2.py
```

It passes when the prepared arrays have shape `(*, 128, 6)`, contain only
finite values, share no recording between splits, the saved Keras model and
confusion-matrix image are substantive files, `metrics.json` contains a real
3 × 3 confusion matrix and per-class recall for all three labels, and test
accuracy is at least 70%.

## Repository structure at the end of Week 2

Items marked *(generated, not committed)* are created on your machine and are
excluded by `.gitignore`.

```text
TinyML-Gesture-Recognition/
├── .gitignore
├── NOTES.md                              # now holds accuracy and experiment entries
├── PROGRESS.md
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
├── week2-model-training/                 # ACTIVE THIS WEEK
│   ├── README.md                         # this overview
│   ├── learning.md                       # the complete Week 2 lesson
│   ├── prepare_data.py
│   ├── train.ipynb
│   ├── validate_week2.py
│   ├── prepared_data.npz                      # (generated, not committed)
│   └── models/
│       ├── .gitkeep
│       ├── gesture_model.keras                # (generated, not committed)
│       ├── metrics.json                       # (generated, not committed)
│       └── confusion_matrix.png               # (generated, not committed)
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

Compared with the end of Week 1, four new files exist:
`week2-model-training/prepared_data.npz`, `models/gesture_model.keras`,
`models/metrics.json`, and `models/confusion_matrix.png`. All four stay on
your machine.

## Next step

Open [`learning.md`](learning.md) and start Day 1. When
`validate_week2.py` prints `PASSED`, tick the Week 2 boxes in
[`../PROGRESS.md`](../PROGRESS.md) and continue to
[Week 3](../week3-deployment/README.md).
