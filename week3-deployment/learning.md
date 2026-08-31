# Week 3: Deployment

## Goals

This week you will compress the trained model, run it on live sensor windows,
measure the deployment artifact, and capture an honest portfolio demo. You can
deploy on the Arduino or use the included laptop fallback; both execute the same
quantized model.

## Concepts

### Inference and deployment

**Inference** means applying a trained model to new input. **Deployment** means
packaging that model with the sensor and application code where people will use
it. Training needs gradients and many examples; inference needs only one input,
the saved weights, and a small set of mathematical operations.

The live pipeline must exactly repeat training preparation: feature order is
`ax, ay, az, gx, gy, gz`, a window has 128 samples, and values use the saved
training mean and standard deviation. A **preprocessing mismatch** can ruin a
good model without producing an obvious software error.

### Quantization

The Keras model stores most numbers as 32-bit floating-point values.
**Quantization** maps them to 8-bit integers. A scale and **zero point** describe
the mapping between real and integer values. A **representative dataset** shows
the converter the range of typical training inputs. Full integer quantization
makes model storage smaller and arithmetic suitable for a microcontroller.

Quantization can slightly alter predictions. File-size reduction does not prove
accuracy, so retain the Week 2 test measurement and manually compare live
behavior.

### TensorFlow Lite and memory

TensorFlow Lite is an inference format designed for constrained devices.
TensorFlow Lite Micro is its microcontroller runtime. A **tensor arena** is one
fixed memory region reused for inputs, outputs, and intermediate calculations.
Fixed allocation avoids requesting unpredictable memory while the device runs.

The model is converted into a C byte array because a basic Arduino sketch has
no filesystem. `alignas(16)` places that array and arena on boundaries expected
by efficient processors.

### Confidence and unknown motion

Softmax's largest output is called confidence here, but it is not a guarantee.
This model learned only three classes and was never taught a "no gesture"
class. The application prints `uncertain` below 70%, but unfamiliar motion can
still receive high confidence. This limitation belongs in the demo and
portfolio description.

## Hands-on

### Day 1 — Quantize

Confirm the Week 2 validator passes, then run:

```bash
python week3-deployment/quantize.py
```

It creates `models/gesture_model_int8.tflite` and
`arduino/gesture_infer/model_data.h`. Read the printed float and quantized file
sizes. The script also opens the converted model and proves its input and output
are both signed 8-bit integers.

**Try it yourself 1:** calculate the percentage size reduction yourself and
record it in `NOTES.md`.

**Definition of done**

- [ ] Conversion completes without fallback operations.
- [ ] The `.tflite` file and generated header exist.
- [ ] The quantized file is smaller than the Keras file.
- [ ] Model size and reduction are recorded.

### Day 2 — Choose one live path

#### Path A: inference on Arduino

In Arduino IDE Library Manager, install `Chirale_TensorFlowLite` and confirm
`Arduino_BMI270_BMM150` remains installed. Open
`arduino/gesture_infer/gesture_infer.ino`. Its generated `model_data.h` must be
beside the sketch. Select the Nano 33 BLE board and port, upload, and open Serial
Monitor at 115200 baud.

If compilation fails, copy the complete first error into `NOTES.md` before
changing anything. Library versions occasionally change; use Path B rather
than losing the week to tool installation.

#### Path B: laptop fallback

Upload Week 1's `stream_imu.ino`, close Serial Monitor, and run:

```bash
python week3-deployment/live_infer.py --port /dev/ttyACM0
```

Replace the port as before. This is a genuine deployment fallback: sensor data
is live and the quantized model performs inference, but the laptop provides the
compute and memory.

**Definition of done**

- [ ] Exactly one program owns the serial port.
- [ ] The selected path reports that it is ready.
- [ ] One 128-sample window produces a label and confidence.
- [ ] The chosen deployment location is recorded honestly.

### Day 3 — Test behavior

Use the same board orientation as Week 1. For each label, perform ten separate
2.5-second gestures and tally predicted labels. Then make ten still or unrelated
motions. Record the tally, including failures. These observations are a small
**field test**, not a replacement for the held-out test accuracy.

**Try it yourself 2:** run with laptop `--threshold 0.85` (or change
`kConfidenceThreshold` to `0.85f` on Arduino). Compare uncertain results, then
restore 0.70. A higher threshold usually rejects more correct and incorrect
predictions.

**Definition of done**

- [ ] Ten live trials were completed per known gesture.
- [ ] Stillness or unrelated movement was tested.
- [ ] Failures and uncertain results were preserved.
- [ ] Orientation and window timing matched collection.

### Day 4 — Explain limitations and capture a demo

Write down who supplied training data, the three supported labels, test-set
size, accuracy, model size, and known confusions. Do not claim the system works
for every user or movement.

Capture a short GIF or video in `demo/`. Keep the prediction output visible,
show each gesture at least once, and include one continuous take rather than
editing together only successes. Add a short caption to the root README if you
plan to publish it.

**Try it yourself 3:** ask someone whose movements were not in training data to
try the demo with consent. Record the result as an observation, not a new test
accuracy.

**Definition of done**

- [ ] Demo visibly connects motion to live output.
- [ ] Accuracy and model size use measured numbers.
- [ ] Unsupported motion and user-generalization limits are stated.
- [ ] No raw personal data is committed without consent.

### Day 5 — Prove completion

```bash
python week3-deployment/validate_week3.py
```

The validator parses the real TFLite signature, compares file sizes, checks
every byte in the C header, verifies matching normalization and labels, retains
the measured accuracy requirement, and requires a substantive demo artifact.

**Definition of done**

- [ ] Validation reports `PASSED`.
- [ ] All three weekly validators pass from the repository root.
- [ ] `PROGRESS.md` is complete.
- [ ] Final results and limitations are in `NOTES.md`.

## Code walkthrough

`quantize.py` loads the final Keras model and yields training samples to the
converter as representative inputs. Integer-only operation selection plus int8
input and output prevents accidental floating-point boundaries. It writes the
flatbuffer unchanged, turns every byte into hexadecimal C data, and carries
normalization values and class order into the same generated header.

`gesture_infer.ino` initializes the sensor and interpreter once. At 50 Hz it
normalizes six readings and quantizes each into the input tensor. After 128
samples, it invokes the model, converts output integers back to probabilities,
prints the strongest acceptable class, and begins a new window.

`live_infer.py` performs the same sequence using a deque, a fixed-length queue.
It reads model quantization metadata instead of assuming a scale. The fallback
supports TensorFlow's interpreter and the smaller `tflite-runtime` interpreter.

`validate_week3.py` examines generated bytes and metrics without requiring
TensorFlow. It also checks proof that the system was demonstrated rather than
merely converted.

## Validation

Automated checks prove artifact integrity, size reduction, and retained Week 2
quality evidence. Your ten-trial tally proves only how the current setup behaved
in that session. Report both, with their different meanings.

## Exit criteria

- [ ] All Day 1–5 definitions of done are checked.
- [ ] You can explain inference, quantization, scale, zero point, and tensor arena.
- [ ] Live inference works on the Arduino or documented laptop fallback.
- [ ] Size, accuracy, field failures, and limits are presented honestly.
- [ ] All three validators pass without weakening their requirements.
