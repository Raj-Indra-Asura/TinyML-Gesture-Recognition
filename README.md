# TinyML Gesture Recognition

A three-week, learn-by-building course that takes you from raw motion data to a
gesture classifier running on an Arduino Nano 33 BLE Sense Rev2. You need basic
Python knowledge, but no embedded systems or machine-learning experience.

## What you will build

The finished system reads six motion values (three acceleration axes and three
rotation axes), recognizes `circle`, `left_right`, and `up_down`, and reports
its prediction over USB. Week 3 also includes laptop inference if the
TensorFlow Lite Micro Arduino library is unavailable.

## Hardware and software

- Arduino Nano 33 BLE Sense Rev2 and a data-capable USB cable
- A laptop with Arduino IDE 2
- Python 3.10 or newer
- Python packages: `pyserial`, `numpy`, `matplotlib`, `tensorflow`, and
  `jupyter`

Create an environment and install the packages:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install pyserial numpy matplotlib tensorflow jupyter
```

In Arduino IDE, install the **Arduino_BMI270_BMM150** library in Week 1. Install
**Chirale_TensorFlowLite** before the optional on-device deployment in Week 3.

## Course index

Each week has two documents. Read the week's `README.md` first for the
overview, the deliverables, and the repository structure at the end of that
week. Then work through its `learning.md`, which is a complete, self-contained
lesson: every instruction, explanation, command, expected output, code
walkthrough, and troubleshooting table you need. You should not need any other
book, video, or website.

The project is built sequentially, each week completing a defined share of it:

| Week | Overview | Full lesson | Delivers | Cumulative completion |
| --- | --- | --- | --- | --- |
| 1 — Data Foundations | [README](week1-data-foundations/README.md) | [learning.md](week1-data-foundations/learning.md) | Board brought up, sensor streaming, 45+ labeled recordings validated | **35%** |
| 2 — Model Training | [README](week2-model-training/README.md) | [learning.md](week2-model-training/learning.md) | Fixed-size normalized examples, leak-free splits, trained model, measured accuracy | **75%** |
| 3 — Deployment | [README](week3-deployment/README.md) | [learning.md](week3-deployment/learning.md) | Quantized model, C header, live inference, field test, demo, limitations | **100%** |

Follow the weeks in order; each one depends on the artifacts the previous week
produced. Read each concept section before running its code. Use
[PROGRESS.md](PROGRESS.md) as the course checklist and [NOTES.md](NOTES.md) as
your experiment log. Paths in commands assume you are at the repository root.

## Repository layout

These are the files tracked in Git. Each week's `README.md` also shows the full
structure, including the files you generate locally, as it will look at the end
of that week.

```text
TinyML-Gesture-Recognition/
├── README.md                     # this page
├── PROGRESS.md                   # course checklist
├── NOTES.md                      # your experiment log
├── week1-data-foundations/
│   ├── README.md                 # Week 1 overview and structure
│   ├── learning.md               # Week 1 complete lesson
│   ├── collect_data.py
│   ├── validate_week1.py
│   ├── arduino/stream_imu/stream_imu.ino
│   └── data/.gitkeep
├── week2-model-training/
│   ├── README.md                 # Week 2 overview and structure
│   ├── learning.md               # Week 2 complete lesson
│   ├── prepare_data.py
│   ├── train.ipynb
│   ├── validate_week2.py
│   └── models/.gitkeep
└── week3-deployment/
    ├── README.md                 # Week 3 overview and structure
    ├── learning.md               # Week 3 complete lesson
    ├── quantize.py
    ├── live_infer.py
    ├── validate_week3.py
    ├── arduino/gesture_infer/gesture_infer.ino
    └── demo/README.md
```

## Data and generated artifacts

Your raw recordings and trained model files are intentionally not committed.
They describe your movements and can be recreated. Placeholder files keep the
required directories in Git; the validation scripts inspect your real local
artifacts.

## Completion

You are finished when all three commands pass:

```bash
python week1-data-foundations/validate_week1.py
python week2-model-training/validate_week2.py
python week3-deployment/validate_week3.py
```

Record the exact results in `NOTES.md` and add a short demo to
`week3-deployment/demo/` before presenting the project.
