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

1. [Week 1 — Data Foundations](week1-data-foundations/learning.md): connect the
   board, understand its motion sensor, and collect trustworthy examples.
2. [Week 2 — Model Training](week2-model-training/learning.md): turn recordings
   into fixed-size examples, train a neural network, and measure it honestly.
3. [Week 3 — Deployment](week3-deployment/learning.md): compress the model,
   run live predictions, and produce a portfolio demo.

Follow the weeks in order. Read each concept section before running its code.
Use [PROGRESS.md](PROGRESS.md) as the course checklist and [NOTES.md](NOTES.md)
as your experiment log. Paths in commands assume you are at the repository
root.

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
