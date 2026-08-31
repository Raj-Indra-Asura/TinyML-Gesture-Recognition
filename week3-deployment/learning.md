# Week 3: Deployment

**Project progress this week: 75% → 100%.**
Start here after reading [`README.md`](README.md).

This file is the only resource you need for Week 3. It explains quantization,
TensorFlow Lite, and on-device inference from first principles, gives the exact
commands and expected output, walks through every line of code you run, and
tells you what to do when a tool fights back. You do not need any other
resource.

---

## Table of contents

1. [Where you are and what "deployment" means](#1-where-you-are-and-what-deployment-means)
2. [Inference versus training](#2-inference-versus-training)
3. [Quantization explained from scratch](#3-quantization-explained-from-scratch)
4. [TensorFlow Lite, C arrays, and the tensor arena](#4-tensorflow-lite-c-arrays-and-the-tensor-arena)
5. [Confidence, thresholds, and unknown motion](#5-confidence-thresholds-and-unknown-motion)
6. [Day 1 — Quantize](#day-1--quantize)
7. [Day 2 — Choose one live path](#day-2--choose-one-live-path)
8. [Day 3 — Test behavior](#day-3--test-behavior)
9. [Day 4 — Explain limitations and capture a demo](#day-4--explain-limitations-and-capture-a-demo)
10. [Day 5 — Prove completion](#day-5--prove-completion)
11. [Code walkthrough](#11-code-walkthrough)
12. [Troubleshooting](#12-troubleshooting)
13. [Self-check questions and answers](#13-self-check-questions-and-answers)
14. [Glossary](#14-glossary)
15. [Exit criteria](#15-exit-criteria)

---

## 1. Where you are and what "deployment" means

### Prerequisite

This must print `PASSED` before you start:

```bash
python week2-model-training/validate_week2.py
```

Week 3 converts Week 2's exact saved model. Converting a model you have since
changed produces artifacts that do not match your recorded measurements.

### What is still missing

You have a trained model that scores well on saved recordings. Three things
stand between that and a working product:

1. **Size.** The Keras file is well over 100 KiB of 32-bit floating-point
   numbers. The board has limited flash and RAM, and floating-point arithmetic
   is comparatively slow on it.
2. **Live input.** Nothing yet feeds fresh sensor data into the model as it
   arrives.
3. **Honest reporting.** Nothing yet states what the system cannot do.

Week 3 solves all three, and that is the final 25% of the project.

| Week | Delivers | Cumulative |
| --- | --- | --- |
| 1 | Validated dataset | 35% |
| 2 | Trained model and measured accuracy | 75% |
| **3 (this week)** | Quantized model, C header, live inference, field test, demo, limitations | **100%** |

### Activate your environment

```bash
cd TinyML-Gesture-Recognition
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
```

---

## 2. Inference versus training

**Inference** means using a trained model to answer a question about new input.
**Deployment** means packaging that model together with the sensor and the
application code, in the place where people will actually use it.

The two jobs have very different needs:

| | Training (Week 2) | Inference (Week 3) |
| --- | --- | --- |
| Runs how often | Once, offline | Continuously, live |
| Needs the whole dataset | Yes | No — one window at a time |
| Needs gradients and an optimizer | Yes | No |
| Needs to change weights | Yes | No — weights are frozen |
| Typical hardware | Laptop or server | Microcontroller |

Because inference never changes the weights, everything associated with
learning can be thrown away. What remains is a fixed set of numbers and a short
sequence of multiplications and additions — small enough to fit on a chip with
a few hundred kilobytes of memory.

### The rule that breaks more deployments than anything else

**Live preprocessing must match training preprocessing exactly.** Concretely:

- Feature order must be `ax, ay, az, gx, gy, gz`.
- A window must contain exactly 128 samples.
- Values must be normalized with the *same* six means and six standard
  deviations that `prepare_data.py` computed from the training split.
- The board must be held in the same orientation as in Week 1.

Get any of these wrong and there is **no error message**. The model runs
happily and outputs confident nonsense. This is called a **preprocessing
mismatch**, and it is why `quantize.py` copies the normalization values
directly into the generated C header instead of asking you to retype them.

---

## 3. Quantization explained from scratch

### 3.1 What a float is, and why it is expensive

Your Keras model stores each weight as a **32-bit floating-point number**
(float32): four bytes, capable of representing a huge range with fine
precision. That precision is useful while training, where weights change by
tiny amounts. Once training is finished it is largely wasted.

### 3.2 The idea

**Quantization** replaces each float32 with an **8-bit integer** (int8): one
byte, holding a whole number from −128 to 127. That is a 4× reduction in size,
and integer arithmetic is faster and cheaper on a microcontroller.

The mapping between the two worlds uses two extra numbers per tensor:

- **Scale**: how much one integer step is worth in real units.
- **Zero point**: which integer represents real value 0.0.

```
real  ≈ (integer − zero_point) × scale
integer ≈ round(real / scale) + zero_point
```

### 3.3 A worked example

Suppose a tensor's real values all lie between −1.0 and +1.0, and you want to
store them in int8 (−128 to 127).

```
range covered = 1.0 − (−1.0) = 2.0
steps available = 255
scale = 2.0 / 255 ≈ 0.00784
zero_point = 0        (because the range is symmetric here)
```

Encode `0.5`:  `round(0.5 / 0.00784) + 0 = round(63.8) = 64`
Decode `64`:   `(64 − 0) × 0.00784 = 0.502`

You stored 0.5 and got back 0.502. That small difference is **quantization
error**. It exists for every weight, and it is the price of the 4× saving.

### 3.4 The representative dataset

To choose a good scale, the converter needs to know the realistic range of
values flowing through the network. It cannot guess. So you hand it a
**representative dataset**: a sample of genuine training inputs. The converter
runs them through, observes the actual minimum and maximum at each tensor, and
picks scales that cover those ranges without wasting resolution.

This is why `quantize.py` needs `prepared_data.npz` and not just the model. If
you gave it random noise instead, the scales would be chosen for value ranges
that never occur, and accuracy would suffer badly.

### 3.5 Full integer quantization

This project uses **full integer quantization**: weights, activations, the
model input, and the model output are all int8. The alternative — leaving the
input and output as floats — creates float-to-int conversion points that some
microcontroller runtimes cannot execute at all. Going fully integer is what
makes the model deployable, and `quantize.py` verifies it by checking that the
converted model's input and output dtypes really are `int8`.

### 3.6 What quantization does *not* prove

A smaller file is not a better model. Quantization can slightly change
predictions, usually by a negligible amount but occasionally near a decision
boundary. So:

- Keep reporting **Week 2's measured test accuracy** as your accuracy figure.
- Report the **size reduction** separately as a size result.
- Treat this week's live trials as a qualitative field check, not a new
  accuracy number.

---

## 4. TensorFlow Lite, C arrays, and the tensor arena

### 4.1 TensorFlow Lite and TensorFlow Lite Micro

**TensorFlow Lite** is a compact file format plus a small interpreter for
running trained models on phones and embedded devices. A `.tflite` file is a
**flatbuffer** — a binary layout that can be read directly from memory without
being unpacked first, which matters when you have very little RAM. Its bytes 4
through 7 spell `TFL3`, which is why `validate_week3.py` checks for that
signature.

**TensorFlow Lite Micro** is the even smaller variant for microcontrollers. It
has no operating system dependencies, no file system, and no dynamic memory
allocation.

### 4.2 Why the model becomes a C array

The Arduino has no filesystem, so there is nowhere to put a `.tflite` file for
the program to open. Instead, the model's bytes are written directly into the
program's source code as a C array:

```cpp
alignas(16) const unsigned char g_model[] = {
  0x1c, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33, ...
};
constexpr unsigned int g_model_len = 38912;
```

`const` places it in flash memory (of which there is a lot) rather than RAM (of
which there is little). `alignas(16)` forces the array to begin at an address
that is a multiple of 16, because the processor reads aligned memory faster and
some operations require it.

`quantize.py` generates this file, `model_data.h`, automatically. Never edit it
by hand.

### 4.3 The tensor arena

While the model runs it needs scratch space: somewhere for the input, the
output, and the intermediate results between layers. On a laptop the program
would just allocate memory as needed. On a microcontroller, allocating
unpredictably at run time risks running out of memory hours into operation.

So TensorFlow Lite Micro uses a **tensor arena**: one fixed block of memory,
reserved up front, that it reuses for everything.

```cpp
constexpr int kTensorArenaSize = 120 * 1024;
alignas(16) byte tensorArena[kTensorArenaSize];
```

If the arena is too small, `AllocateTensors()` fails immediately at startup and
the sketch prints an error — a predictable failure at a predictable moment,
which is exactly what you want on a device that might run unattended.

### 4.4 The windowing loop

Live inference is not one prediction; it is a repeating cycle:

1. Read one sample (six values) every 20 ms.
2. Normalize it, quantize it, and store it in the input tensor.
3. When 128 samples have accumulated — about 2.56 seconds — run the model.
4. Print the result, reset the counter, and begin the next window.

128 samples at 50 Hz is 2.56 seconds, which is why the instructions ask for
roughly 2.5-second gestures: the window should contain one complete gesture and
little else.

---

## 5. Confidence, thresholds, and unknown motion

The model's softmax output is three numbers summing to 1, such as
`0.94, 0.05, 0.01`. The largest one is reported as **confidence**.

Two honest cautions:

1. **Softmax always sums to 1.** If you hold the board perfectly still, the
   model still must distribute all its probability across the three gestures.
   It has no fourth option.
2. **The model was never taught "no gesture".** Every training example was one
   of three deliberate movements. An unfamiliar movement can land near one
   class's region and receive high confidence.

The **confidence threshold** softens this. Both programs print `uncertain` when
the best probability is below 0.70:

```python
prediction = labels[best] if confidence >= args.threshold else "uncertain"
```

Raising the threshold rejects more wrong predictions *and* more right ones;
lowering it does the reverse. There is no setting that fixes the underlying
gap. The genuine fix would be to collect an `idle` class in Week 1 and train
four classes — a good extension after you finish, and a good thing to say in an
interview about this project.

This limitation belongs in your demo caption and in `NOTES.md`. Stating it is a
sign of engineering maturity, not a weakness.

---

## Day 1 — Quantize

**Objective:** produce the deployable `.tflite` model and its C header.

### Run it

```bash
python week3-deployment/quantize.py
```

### Expected output

```
Saved week2-model-training/models/gesture_model_int8.tflite (38.0 KiB)
Float Keras file: 148.6 KiB
Saved Arduino header to week3-deployment/arduino/gesture_infer/model_data.h
Input quantization (scale, zero point): (0.0392156, -1)
```

Your numbers will differ. What matters is that the `.tflite` file is clearly
smaller than the Keras file, and that the script did not report a failure.

### What was created, and where

| File | Location | Why there |
| --- | --- | --- |
| `gesture_model_int8.tflite` | `week2-model-training/models/` | Beside the Keras model it was converted from |
| `model_data.h` | `week3-deployment/arduino/gesture_infer/` | Beside the sketch that `#include`s it |

`model_data.h` contains four things: the model bytes, the six normalization
means, the six normalization standard deviations, and the label list in the
correct order. Bundling all four together is what prevents a preprocessing
mismatch.

### Try it yourself 1

Compute the size reduction yourself and record it in `NOTES.md`:

```
reduction = (1 − quantized_size / float_size) × 100%
```

With the example numbers above: `(1 − 38.0 / 148.6) × 100 ≈ 74.4%`.

The size roughly follows the arithmetic: about 37 000 parameters at four bytes
each is close to 148 KiB, and the same parameters at one byte each is close to
37 KiB, plus a few kilobytes of model structure. Seeing your own numbers land
near that estimate is a good sign that the conversion did what you expect.

### Definition of done

- [ ] Conversion completes without fallback operations.
- [ ] The `.tflite` file and generated header exist.
- [ ] The quantized file is smaller than the Keras file.
- [ ] Model size and reduction are recorded in `NOTES.md`.

---

## Day 2 — Choose one live path

**Objective:** get one live prediction from real motion.

Both paths run the *same* quantized model on *live* sensor data. Path A runs it
on the microcontroller; Path B runs it on your laptop while the board streams
sensor readings. Pick one. Say which one you used.

> **Do not lose the week to library installation.** If Path A does not compile
> within a reasonable effort, copy the complete first error into `NOTES.md`,
> switch to Path B, and move on. Recording a tool failure and choosing a
> working alternative is a legitimate engineering result.

### Path A — inference on the Arduino

1. In Arduino IDE, **Tools → Manage Libraries**, install
   `Chirale_TensorFlowLite`. Confirm `Arduino_BMI270_BMM150` is still
   installed.
2. Open `week3-deployment/arduino/gesture_infer/gesture_infer.ino`. The
   generated `model_data.h` must be in the same folder — `quantize.py` already
   put it there. In the IDE it appears as a second tab.
3. Select **Arduino Nano 33 BLE** and your port, then **Upload**. This sketch
   is much larger than Week 1's, so compilation takes noticeably longer.
4. Open **Serial Monitor** at 115200 baud. You should see:
   ```
   Ready. Perform one gesture for about 2.5 seconds.
   ```
5. Perform a gesture. After about 2.56 seconds a line appears:
   ```
   circle, confidence=0.942
   ```

### Path B — laptop fallback

1. Upload Week 1's `stream_imu.ino` to the board (it just streams sensor data).
2. **Close Serial Monitor** — Python needs the port.
3. Run:
   ```bash
   python week3-deployment/live_infer.py --port /dev/ttyACM0
   ```
4. You should see:
   ```
   Hold still while serial starts. Press Ctrl+C to stop.
         circle  confidence=94.2%
   ```
5. Press **Ctrl+C** to stop.

Useful options: `--threshold` (default 0.70), `--baud`, `--model`, `--data`.

### Why Path B is a real result

The sensor data is live, and the quantized int8 model performs the inference.
The only difference is where the computation happens. Describe it accurately —
"quantized model, live sensor input, inference on laptop" — and it is a
completely honest deliverable.

### Definition of done

- [ ] Exactly one program owns the serial port.
- [ ] The selected path reports that it is ready.
- [ ] One 128-sample window produces a label and a confidence.
- [ ] The chosen deployment location is recorded honestly in `NOTES.md`.

---

## Day 3 — Test behavior

**Objective:** find out how the system behaves in the real world, including
when it fails.

### The procedure

Use the same board orientation and gesture style as Week 1.

1. For each of the three labels, perform **ten** separate gestures of about 2.5
   seconds each, pausing between them. Write down what the system predicted
   each time.
2. Then perform **ten** movements it was never trained on: hold perfectly
   still, wave randomly, draw a figure-eight, set the board down.
3. Record every result — especially the wrong ones.

### A tally you can copy into `NOTES.md`

```
Live trials (Path B, threshold 0.70)

circle      : circle 8  | left_right 1 | up_down 0 | uncertain 1
left_right  : circle 0  | left_right 7 | up_down 2 | uncertain 1
up_down     : circle 0  | left_right 1 | up_down 8 | uncertain 1
still/other : circle 3  | left_right 2 | up_down 1 | uncertain 4
```

That last row is the most interesting one, and the one most students omit. It
shows what happens with input the model was never designed to handle.

### What this measurement is and is not

This is a **field test**: 30 trials, one person, one session, one setup. It
describes today's behavior. It is *not* a replacement for the Week 2 test
accuracy, which was measured on held-out data under controlled conditions.
Report both, and say clearly which is which.

If live results are much worse than Week 2's accuracy, suspect a preprocessing
or timing mismatch before you suspect the model:

- Are you holding the board in the same orientation as during collection?
- Is your live gesture about 2.5 seconds, matching the window?
- Did the window boundary cut your gesture in half? (Windows do not wait for
  you to start; pause between attempts and start right after a result prints.)
- Is `prepared_data.npz` still the file the model was trained with?

### Try it yourself 2 — the threshold trade-off

Run with a stricter threshold:

```bash
python week3-deployment/live_infer.py --port /dev/ttyACM0 --threshold 0.85
```

(On Arduino, change `kConfidenceThreshold` to `0.85f` and re-upload.)

Repeat a few trials of each gesture plus a few still ones. You should see more
`uncertain` in *both* categories: the threshold cannot tell a wrong confident
answer from a right one, it only filters by confidence. Then restore 0.70.

Write one sentence in `NOTES.md` about what you gave up and what you gained.

### Definition of done

- [ ] Ten live trials were completed per known gesture.
- [ ] Stillness or unrelated movement was tested.
- [ ] Failures and uncertain results were preserved, not deleted.
- [ ] Orientation and window timing matched collection.

---

## Day 4 — Explain limitations and capture a demo

**Objective:** make the project presentable and truthful.

### Write the limitations section

In `NOTES.md`, record:

| Item | Example |
| --- | --- |
| Who supplied training data | One person, single session, right hand |
| Supported labels | `circle`, `left_right`, `up_down` — and nothing else |
| Test set size | 9 held-out recordings |
| Measured accuracy | Week 2's exact number, with that test size |
| Model size | Float and quantized sizes, plus the reduction |
| Deployment location | Arduino, or laptop fallback |
| Known confusions | Largest off-diagonal cell from the confusion matrix |
| Behavior on unknown motion | Your Day 3 still/other row |

Claims to avoid: "works for anyone", "recognizes any gesture", "99% accurate".
Each is refuted by your own measurements. A precise, modest claim backed by
numbers is far more impressive than an inflated one.

### Capture the demo

Record a short GIF or video into `week3-deployment/demo/`. The validator
accepts `.gif`, `.mp4`, `.mov`, or `.webm` of at least 1000 bytes.

Requirements:

- The prediction output stays visible on screen.
- Each of the three gestures appears at least once.
- Include **one continuous take** rather than a montage of successes only.
- If a prediction is wrong or `uncertain` during the take, leave it in and
  mention it in the caption. A demo that only shows successes is not evidence.

Practical capture options: any screen recorder on your operating system; place
the terminal or Serial Monitor window and your hand in the same frame if you
can.

Note that `demo/` is in `.gitignore` (except its `README.md`), so your recording
stays local unless you deliberately publish it.

### Try it yourself 3

With their consent, ask someone whose movements were not in the training data
to try the system. Almost certainly it performs worse for them — the training
data described *your* movements. Record this as an observation about
generalization, not as a new accuracy figure.

### Definition of done

- [ ] Demo visibly connects motion to live output.
- [ ] Accuracy and model size use measured numbers.
- [ ] Unsupported motion and user-generalization limits are stated.
- [ ] No raw personal data is committed without consent.

---

## Day 5 — Prove completion

**Objective:** finish the project at 100%.

### Run the Week 3 validator

```bash
python week3-deployment/validate_week3.py
```

### What it checks

| Check | Why |
| --- | --- |
| `.tflite` exists, is > 1000 bytes, and its bytes 4–7 are `TFL3` | It is a real TFLite flatbuffer, not an empty or corrupt file |
| The quantized file is smaller than the Keras file | Quantization actually did something |
| `model_data.h` exists | The Arduino path is genuinely buildable |
| `g_model_len` equals the `.tflite` byte count | The header matches the model |
| Hex byte count in the header equals the model length | The array was written completely |
| Header contains `kNormalizationMean`, `kNormalizationStd`, `kGestureLabels` | Preprocessing travels with the model |
| Week 2's `test_accuracy` ≥ 0.70 | Quality evidence is retained |
| `demo/` contains a substantive GIF or video | The system was demonstrated, not merely converted |

### Passing output

```
Week 3 validation PASSED: model is 38.0 KiB
- file-size reduction from Keras model: 74.4%
- measured test accuracy: 88.9%
- demo: gesture_demo.gif
```

### Then run all three

From the repository root:

```bash
python week1-data-foundations/validate_week1.py
python week2-model-training/validate_week2.py
python week3-deployment/validate_week3.py
```

All three passing is the definition of a finished project. Anything less is a
project in progress.

### Definition of done

- [ ] Validation reports `PASSED`.
- [ ] All three weekly validators pass from the repository root.
- [ ] `PROGRESS.md` is complete.
- [ ] Final results and limitations are in `NOTES.md`.

---

## 11. Code walkthrough

### 11.1 `quantize.py`

```python
model = tf.keras.models.load_model(args.model)
with np.load(args.data) as data:
    representative_values = data["x_train"].astype(np.float32)
    mean = data["normalization_mean"].astype(np.float32)
    std = data["normalization_std"].astype(np.float32)
    labels = [str(value) for value in data["labels"]]
```

Loads Week 2's exact model and its exact preparation statistics. Reading the
normalization values from the same `.npz` the model trained with is what
guarantees the deployed pipeline matches the trained one.

```python
def representative_dataset():
    for sample in representative_values:
        yield [sample[np.newaxis, ...]]
```

Section 3.4 in code. It is a **generator**: it produces one training example at
a time rather than building a list, so memory stays flat. `[np.newaxis, ...]`
adds a leading batch dimension, turning `(128, 6)` into `(1, 128, 6)`, because
the converter expects batched input.

```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
```

The four settings that force *full* integer quantization. Restricting
`supported_ops` to the int8 builtin set means conversion **fails loudly** if any
operation cannot be expressed in integers, rather than silently inserting a
float operation the microcontroller runtime cannot execute.

```python
interpreter = tf.lite.Interpreter(model_content=model_bytes)
interpreter.allocate_tensors()
...
if input_detail["dtype"] != np.int8 or output_detail["dtype"] != np.int8:
    print("Conversion did not create int8 input and output tensors.")
    return 1
```

The script does not trust the converter's settings; it opens the result and
verifies the tensor types. Verify, then claim.

```python
def format_float_array(name: str, values: np.ndarray) -> str:
    body = ", ".join(f"{float(value):.9e}f" for value in values)
    return f"constexpr float {name}[] = {{{body}}};"
```

Writes the normalization values as valid C float literals. The `f` suffix marks
them as `float` rather than `double`, and 9 significant digits in scientific
notation preserves the value precisely.

```python
for offset in range(0, len(model_bytes), 12):
    chunk = model_bytes[offset : offset + 12]
    byte_lines.append("  " + ", ".join(f"0x{value:02x}" for value in chunk) + ",")
```

Turns the flatbuffer into hexadecimal C data, 12 bytes per line so the file
stays readable. The model bytes themselves are written to disk unchanged; only
this text representation is reformatted.

### 11.2 `gesture_infer.ino`

```cpp
constexpr int kTensorArenaSize = 120 * 1024;
alignas(16) byte tensorArena[kTensorArenaSize];
```

Section 4.3: one fixed memory block, aligned for efficiency.

```cpp
const tflite::Model* model = tflite::GetModel(g_model);
if (model->version() != TFLITE_SCHEMA_VERSION) {
```

Points the interpreter at the array from `model_data.h` and checks that the
file format version matches the installed library. A version mismatch is caught
at startup with a clear message rather than as mysterious wrong output.

```cpp
static tflite::AllOpsResolver resolver;
static tflite::MicroInterpreter staticInterpreter(
    model, resolver, tensorArena, kTensorArenaSize);
```

A **resolver** supplies the implementations of the operations the model uses.
`AllOpsResolver` includes every available operation — simple and reliable,
at the cost of some program size. `static` gives these objects a fixed lifetime
for the whole program, avoiding dynamic allocation.

```cpp
if (interpreter->AllocateTensors() != kTfLiteOk) {
  Serial.println("ERROR: tensor arena is too small");
```

Where an undersized arena is detected: at startup, deterministically.

```cpp
if (input->type != kTfLiteInt8 || output->type != kTfLiteInt8) {
  Serial.println("ERROR: expected a fully int8 model");
```

The sketch's own check that it was given the fully quantized model it expects.

```cpp
int8_t quantizeInput(float value) {
  const float scale = input->params.scale;
  const int32_t zeroPoint = input->params.zero_point;
  long quantized = lroundf(value / scale) + zeroPoint;
  quantized = constrain(quantized, -128, 127);
  return static_cast<int8_t>(quantized);
}
```

Section 3.2's encode formula, in C. Note that the scale and zero point are read
from the *model* rather than hard-coded, and `constrain` clamps to the int8
range so an extreme reading cannot wrap around to a wildly wrong value.

```cpp
const float normalized =
    (values[feature] - kNormalizationMean[feature]) /
    kNormalizationStd[feature];
const int offset = sampleIndex * kFeatureCount + feature;
input->data.int8[offset] = quantizeInput(normalized);
```

Normalize with the training statistics, then quantize, then store. The `offset`
arithmetic lays the 128 × 6 window out as one flat row in memory, in the same
order the model expects.

```cpp
++sampleIndex;
if (sampleIndex < kWindowSamples) {
  return;
}
```

Fill the window one sample at a time, and only run the model when it is
complete.

```cpp
const float probability =
    (output->data.int8[index] - output->params.zero_point) *
    output->params.scale;
```

Section 3.2's decode formula: integer output back to a probability.

```cpp
if (bestProbability >= kConfidenceThreshold) {
  Serial.print(kGestureLabels[bestIndex]);
} else {
  Serial.print("uncertain");
}
...
sampleIndex = 0;
```

Apply the threshold, print, and reset the counter so the next window begins.

### 11.3 `live_infer.py`

```python
window: deque[np.ndarray] = deque(maxlen=WINDOW_SIZE)
```

A **deque** with `maxlen` is a fixed-length queue: appending to a full one
automatically discards the oldest item. It is the natural structure for a
sliding window of the most recent 128 samples.

```python
try:
    from tensorflow.lite import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
```

Supports both the full TensorFlow install and the much smaller
`tflite-runtime`, so the fallback works on a machine where installing all of
TensorFlow is impractical.

```python
def quantize(values, detail):
    dtype = detail["dtype"]
    if np.issubdtype(dtype, np.floating):
        return values.astype(dtype)
    scale, zero_point = detail["quantization"]
    if scale <= 0:
        raise RuntimeError("Model input has invalid quantization metadata")
    limits = np.iinfo(dtype)
    transformed = np.rint(values / scale + zero_point)
    return np.clip(transformed, limits.min, limits.max).astype(dtype)
```

The same encode formula as the sketch, reading scale and zero point from the
model's metadata rather than assuming them, and clipping to the integer type's
real limits. A non-positive scale would indicate a corrupt model, so it raises
instead of producing garbage.

```python
normalized = (np.stack(window) - mean) / std
input_value = quantize(normalized[np.newaxis, ...], input_detail)
interpreter.set_tensor(input_detail["index"], input_value)
interpreter.invoke()
probabilities = dequantize(
    interpreter.get_tensor(output_detail["index"])[0], output_detail
)
```

The full inference sequence: normalize with training statistics, quantize, set
the input tensor, run, read the output, convert back to probabilities. Exactly
the same steps as the Arduino sketch, which is the point.

```python
window.clear()
```

Clears the window after a prediction, so windows do not overlap. Overlapping
windows would print several predictions per gesture, some containing only part
of the movement.

### 11.4 `validate_week3.py`

```python
if len(model_bytes) < 1000 or model_bytes[4:8] != b"TFL3":
    errors.append("quantized model is not a substantive TFLite flatbuffer")
```

Checks the actual file signature rather than the extension. Renaming any file
to `.tflite` will not fool it.

```python
length_match = re.search(r"g_model_len = (\d+);", header)
byte_count = len(re.findall(r"0x[0-9a-fA-F]{2}", header))
if not length_match or int(length_match.group(1)) != len(model_bytes):
    ...
if byte_count != len(model_bytes):
    errors.append("model_data.h byte array is incomplete")
```

Verifies both the declared length *and* the number of hex bytes actually
present. A truncated header — from a failed write or a partial copy — is caught
here rather than as inexplicable behavior on the board.

```python
demo_files = [
    path for path in demo_dir.iterdir()
    if path.suffix.lower() in {".gif", ".mp4", ".mov", ".webm"}
    and path.stat().st_size >= 1000
]
```

Requires evidence that the system was actually demonstrated. Converting a model
is not the same as showing it work.

Note that this validator deliberately does **not** import TensorFlow: it
inspects raw bytes and JSON. That keeps it fast and lets it run in environments
where TensorFlow is not installed.

---

## 12. Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `Missing Week 2 model or prepared data.` | Week 2 artifacts absent | Rerun the Week 2 notebook and `prepare_data.py` |
| `Missing TensorFlow. Run: python -m pip install tensorflow` | Environment not activated, or package missing | Activate `.venv`, then install |
| `TensorFlow Lite conversion failed: ...` | An operation cannot be represented in int8 | Keep the standard `Flatten → Dense → Dense` architecture from Week 2 |
| `Conversion did not create int8 input and output tensors.` | Converter settings were modified | Restore the four converter settings in `quantize.py` |
| Arduino: `model_data.h: No such file or directory` | `quantize.py` not run, or the sketch was moved | Run `quantize.py`; keep the sketch and header in the same folder |
| Arduino: `ERROR: tensor arena is too small` | Model needs more scratch memory | Raise `kTensorArenaSize` in steps; if the board runs out of RAM, use Path B |
| Arduino: `TensorFlowLite.h: No such file` | `Chirale_TensorFlowLite` not installed | Install it, or switch to Path B |
| Arduino sketch too large to upload | Library plus model exceeds flash | Reduce the model (Week 2 Day 4 experiment) or use Path B |
| `Missing quantized model or prepared data. Run quantize.py first.` | Day 1 not completed | Run `quantize.py` |
| `Could not read /dev/ttyACM0` | Wrong port, or Serial Monitor is open | Close Serial Monitor; verify the port |
| Live inference prints nothing | Fewer than 128 samples so far, or the wrong sketch is on the board | Wait ~2.6 s; confirm `stream_imu.ino` is uploaded for Path B |
| Everything predicts one class | Preprocessing or orientation mismatch | Match Week 1's orientation; confirm `prepared_data.npz` matches the model |
| Almost everything is `uncertain` | Threshold too high, or gesture not filling the window | Restore 0.70; perform ~2.5-second gestures |
| Live results far worse than Week 2 accuracy | Window timing or orientation | Work through the Day 3 checklist |
| `demo/ needs a substantive GIF or video` | No demo, or file too small/wrong type | Add a `.gif`, `.mp4`, `.mov`, or `.webm` over 1000 bytes |
| `model_data.h byte array is incomplete` | Header truncated or hand-edited | Rerun `quantize.py`; never edit the header manually |
| `quantized model is not smaller than the Keras model file` | Stale `.tflite` from an earlier model | Delete it and rerun `quantize.py` |

---

## 13. Self-check questions and answers

1. **What do scale and zero point do?**
   They define the mapping between int8 storage and real values:
   `real ≈ (integer − zero_point) × scale`. Scale is the size of one integer
   step; zero point is the integer that means 0.0.

2. **Why does the converter need a representative dataset?**
   To observe the real range of values flowing through the network so it can
   choose scales that cover them. Without it, the ranges would be guesses and
   accuracy would suffer.

3. **Your model is 84% smaller. Is it 84% worse, or better?**
   Neither follows. Size and accuracy are separate measurements. Quantization
   usually changes predictions very little, but the only accuracy figure you
   can honestly cite is Week 2's measured test accuracy.

4. **Why is the model embedded as a C array instead of loaded from a file?**
   The microcontroller has no filesystem. Putting the bytes in the source
   places them in flash memory, where the interpreter can read them directly.

5. **What is a tensor arena and why is it fixed in size?**
   A single pre-allocated block reused for inputs, outputs, and intermediates.
   Fixing it avoids unpredictable run-time allocation, so a memory shortage is
   detected at startup rather than hours later.

6. **You hold the board perfectly still and it confidently reports
   `left_right`. Is it broken?**
   No. It has only three possible answers and was never taught "no gesture".
   Softmax must distribute probability across those three. The threshold helps
   but cannot fully solve it; adding an `idle` class would.

7. **Why must live preprocessing exactly match training preprocessing?**
   The model learned the relationship between normalized inputs and gestures.
   Different feature order, window size, or normalization values produce inputs
   from a different distribution — and no error is raised, just wrong answers.

8. **Is the laptop fallback (Path B) cheating?**
   No, provided you describe it accurately. Live sensor data and the quantized
   model are both real; only the compute location differs. Misreporting it as
   on-device would be the problem.

9. **Why does `validate_week3.py` count the hex bytes in `model_data.h`?**
   To prove the header actually contains the whole model. A truncated array
   would otherwise compile and misbehave in confusing ways.

---

## 14. Glossary

| Term | Meaning |
| --- | --- |
| **`alignas(16)`** | C++ instruction placing data at a 16-byte-aligned address for efficient access |
| **Confidence** | The largest softmax output; not a guarantee of correctness |
| **Confidence threshold** | Minimum confidence for reporting a label instead of `uncertain`; 0.70 here |
| **Deployment** | Packaging a model with sensor and application code where it will be used |
| **Field test** | Live trials in real conditions; descriptive, not a held-out accuracy |
| **Flatbuffer** | Binary format readable directly from memory; `.tflite` uses it |
| **Flash memory** | Non-volatile storage on the board, where `const` data lives |
| **Full integer quantization** | Weights, activations, input, and output all int8 |
| **Generator** | Python function that yields values one at a time instead of building a list |
| **Inference** | Applying a trained model to new input |
| **int8** | 8-bit signed integer, −128 to 127 |
| **float32** | 32-bit floating-point number, four bytes |
| **Preprocessing mismatch** | Live input prepared differently from training input; silently wrong results |
| **Quantization** | Representing float values as integers using a scale and zero point |
| **Quantization error** | Small difference introduced by rounding to integers |
| **Representative dataset** | Sample of real training inputs used to choose quantization scales |
| **Resolver** | Component supplying implementations of the model's operations |
| **Scale** | Real-value size of one integer step |
| **Sliding window** | The most recent N samples, here 128 |
| **TensorFlow Lite** | Compact model format and interpreter for constrained devices |
| **TensorFlow Lite Micro** | Microcontroller runtime for TensorFlow Lite |
| **Tensor arena** | Fixed memory block reused for all inference tensors |
| **Zero point** | Integer value that represents real 0.0 |

---

## 15. Exit criteria

Week 3 is complete — and the project is **100% done** — when all of these are
true:

- [ ] All Day 1–5 definitions of done are checked.
- [ ] You can explain inference, quantization, scale, zero point, representative
      dataset, and tensor arena in your own words.
- [ ] Live inference works on the Arduino or on the documented laptop fallback.
- [ ] Size, accuracy, field-test failures, and limitations are all reported
      honestly and separately.
- [ ] All three validators pass from the repository root without weakening any
      requirement.
- [ ] `PROGRESS.md` is fully ticked and `NOTES.md` records the final results.

### Where to go next, if you want to keep building

Optional extensions, each a genuine improvement rather than busywork:

- Add a fourth `idle` class in Week 1 and retrain, so "no gesture" becomes a
  real answer instead of a threshold hack.
- Collect data from several people, with consent, and measure how much
  cross-user accuracy improves.
- Replace the dense layers with a small 1-D convolutional network and compare
  size and accuracy using the Day 4 controlled-experiment method.
- Add a fourth gesture and observe what happens to the confusion matrix.

None of these are required. The project is finished, and finished honestly, at
the checklist above.
