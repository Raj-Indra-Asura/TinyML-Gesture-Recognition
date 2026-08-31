# Week 2: Model Training

**Project progress this week: 35% → 75%.**
Start here after reading [`README.md`](README.md).

This file is the only resource you need for Week 2. It explains every
machine-learning idea from scratch, gives the exact commands and their expected
output, walks through every line of code you run, and lists what to do when
something fails. You do not need another course, book, or video.

---

## Table of contents

1. [Where you are and what you are about to build](#1-where-you-are-and-what-you-are-about-to-build)
2. [Machine learning from zero](#2-machine-learning-from-zero)
3. [Turning recordings into examples](#3-turning-recordings-into-examples)
4. [How a neural network learns](#4-how-a-neural-network-learns)
5. [Honest evaluation](#5-honest-evaluation)
6. [Day 1 — Prepare examples](#day-1--prepare-examples)
7. [Day 2 — Train](#day-2--train)
8. [Day 3 — Evaluate](#day-3--evaluate)
9. [Day 4 — One controlled experiment](#day-4--one-controlled-experiment)
10. [Day 5 — Prove completion](#day-5--prove-completion)
11. [Code walkthrough](#11-code-walkthrough)
12. [Troubleshooting](#12-troubleshooting)
13. [Self-check questions and answers](#13-self-check-questions-and-answers)
14. [Glossary](#14-glossary)
15. [Exit criteria](#15-exit-criteria)

---

## 1. Where you are and what you are about to build

### Prerequisite

Before you run anything this week, this must print `PASSED`:

```bash
python week1-data-foundations/validate_week1.py
```

Week 2 reads your Week 1 CSV files directly. Starting with an incomplete or
unbalanced dataset guarantees confusing results later, and you will not be able
to tell whether the problem is the data or the model.

### The three things you will produce

| Artifact | What it is |
| --- | --- |
| `prepared_data.npz` | All recordings resampled to a fixed shape, normalized, and split into train / validation / test |
| `models/gesture_model.keras` | The trained neural network |
| `models/metrics.json` + `models/confusion_matrix.png` | The measured evidence of how well it works |

### Week 2's share of the project

| Week | Delivers | Cumulative |
| --- | --- | --- |
| 1 | Validated dataset | 35% |
| **2 (this week)** | Preparation pipeline, trained model, measured accuracy | **75%** |
| 3 | Quantization, live inference, demo, limitations | 100% |

At the end of this week the project *can* recognize gestures. It just cannot
yet do it live, on the board. That is Week 3.

### Activate your environment first

Every command this week assumes you are at the repository root with the virtual
environment active:

```bash
cd TinyML-Gesture-Recognition
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
```

---

## 2. Machine learning from zero

### 2.1 What a model actually is

Normally you write rules: *"if the vertical acceleration swings up and down
more than twice a second, call it `up_down`."* That approach collapses quickly
— everyone waves slightly differently, and you would need dozens of fragile
thresholds.

A **model** takes the opposite approach. It is a mathematical function with
thousands of internal numbers, called **parameters** or **weights**, that start
out random. You show it examples with the right answers attached, and an
algorithm nudges those numbers until the function usually produces the right
answer. Nobody writes the rules; they are discovered.

This is **supervised learning**: learning from labeled examples. Your labels
came from Week 1's filenames.

### 2.2 Features and classes

- **Features** are the measurements the model gets as input. Here there are six
  per moment in time: `ax, ay, az, gx, gy, gz`.
- **Classes** are the possible answers. Here there are exactly three: `circle`,
  `left_right`, `up_down`. Choosing among a fixed set of answers is called
  **classification**.

A crucial consequence: the model has only three possible answers. Show it a
figure-eight, or a still hand, and it must still pick one of the three. It has
no concept of "none of these". Week 3 softens this with a confidence
threshold, but the limitation never fully disappears.

### 2.3 Tensors and shapes

A **tensor** is simply a rectangular block of numbers, described by its
**shape**.

| Shape | Meaning | Example |
| --- | --- | --- |
| `(6,)` | one list of 6 numbers | one sample: `ax…gz` |
| `(128, 6)` | 128 rows × 6 columns | one gesture example |
| `(10, 128, 6)` | 10 of those | a **batch** of 10 examples |

One input to this project's model has shape `(128, 6)`: 128 moments in time,
six features at each moment. Shapes are the single most common source of
errors in machine-learning code, so get used to reading them. `x_train.shape`
tells you exactly what you are holding.

---

## 3. Turning recordings into examples

Your Week 1 recordings are not yet usable as model input. Two problems must be
fixed, and `prepare_data.py` fixes both.

### 3.1 Problem one: recordings have different lengths → resampling

One take has 148 rows, another 152, another 145. USB and sensor timing are not
perfect. But a neural network needs every input to have exactly the same shape.

The fix is **resampling by interpolation**. Think of each of the six columns as
a curve drawn through time. Interpolation reads that curve at 128 evenly spaced
positions from start to finish, and those 128 readings become the example.

A tiny illustration. Suppose one column has 5 values and you want 3:

```
original positions: 0.00  0.25  0.50  0.75  1.00
original values:    10    20    30    40    50
new positions:      0.00        0.50        1.00
new values:         10          30          50
```

When a new position lands between two originals, the value is the proportional
blend of its neighbours. This stretches or compresses a recording in time to a
fixed length. It does not invent a different gesture — a circle resampled to
128 points is still a circle. **Every example becomes `(128, 6)`.**

Why 128 specifically? It is close to the ~150 rows of a three-second recording,
so the stretch is small, and it is a power of two, which is convenient for the
microcontroller in Week 3.

### 3.2 Problem two: features have different scales → normalization

Acceleration values sit roughly between −2 and +2. Gyroscope values can reach
±250. If you feed both to a network unchanged, the gyroscope columns dominate
the arithmetic simply because their numbers are bigger — not because they are
more informative.

**Normalization** puts every feature on a comparable scale. For each of the six
features:

```
normalized = (value − mean) / standard_deviation
```

- The **mean** is the average value of that feature.
- The **standard deviation** measures typical spread around the mean: small
  when values cluster tightly, large when they vary a lot.

After this, each feature is centred near 0 and typically ranges about −3 to +3.

**One rule matters more than the formula:** the mean and standard deviation are
computed from the **training data only**. If you computed them over all data
including the test set, information about the test set would leak into
preparation, and your final score would be slightly dishonest. `prepare_data.py`
does this correctly, and it saves those six means and six standard deviations
into the `.npz` file — Week 3 needs the *identical* numbers for live inference.

### 3.3 The three splits

Your 45 recordings are divided into three groups.

| Split | Share | Role | Analogy |
| --- | --- | --- | --- |
| **Training set** | 70% | The model adjusts its weights on these | Homework |
| **Validation set** | 15% | Watched during training to decide when to stop | Practice quiz |
| **Test set** | 15% | Opened once, at the very end | Final exam |

Why three and not two? If you used the test set to decide when to stop training
or how big the model should be, you would gradually shape the model around the
test set, and it would no longer measure unseen performance. The validation set
absorbs that role so the test set stays untouched.

Two details make the split trustworthy:

- **Whole recordings move together.** A recording is never cut in half with
  part in training and part in test. This is the anti-leakage rule you started
  obeying in Week 1.
- **The split is stratified**, meaning each class is split separately, so all
  three gestures appear in all three groups. Without it, random chance with
  only 45 files could easily produce a test set containing no `up_down` at all.

The random shuffling uses a fixed **seed** of 42. A seed makes "random"
reproducible: run preparation twice and you get exactly the same split, so your
results are comparable from day to day.

---

## 4. How a neural network learns

### 4.1 The architecture used here

```
Input (128, 6)
   ↓ Flatten            →  768 numbers in a row
   ↓ Dense(48) + ReLU   →  48 numbers
   ↓ Dense(3) + softmax →  3 probabilities
```

**Flatten** lays the 128 × 6 grid out as a single row of 768 values. It changes
only the arrangement, never the values.

A **dense layer** (also called fully connected) is the core building block.
Each of its units multiplies every incoming number by a learned weight, adds
them up, adds one learned offset called a **bias**, and passes the result
through an **activation function**:

```
output = activation( w1·x1 + w2·x2 + … + w768·x768 + b )
```

The 48-unit layer therefore learns 48 different weighted combinations of the
motion values — think of each unit as one learned detector.

An **activation function** introduces a bend. Without one, stacking layers
would be pointless: a chain of straight-line functions is still a straight
line, and could never separate curved patterns. This model uses:

- **ReLU** (rectified linear unit): `max(0, x)`. Keeps positives, replaces
  negatives with zero. Simple, fast, and enough to create the needed bend.
- **Softmax** on the final layer: converts three raw scores into three
  positive numbers that add up to exactly 1, so they read as probabilities,
  such as `0.94, 0.05, 0.01`.

Counting the parameters shows why the model is "tiny":
768 × 48 + 48 = 36 912 in the first dense layer, plus 48 × 3 + 3 = 147 in the
second. About 37 000 numbers — small by modern standards, and deliberately so:
it has to fit on a microcontroller in Week 3.

### 4.2 The training loop

1. Take a small group of examples — a **batch** (8 here).
2. Run them through the network. This is the **forward pass**.
3. Measure how wrong the answers are with a **loss function**.
4. Compute which direction each weight should move to reduce the loss. This is
   **backpropagation**, and it is just calculus applied efficiently.
5. Nudge the weights a small step in that direction. The rule for taking the
   step is the **optimizer**.
6. Repeat until every training example has been used once — that is one
   **epoch** — then start again.

The **loss function** here is **cross-entropy**. In words: the penalty is small
when the model gives high probability to the correct class, and grows sharply
as that probability approaches zero. If the true answer is `circle` and the
model said `circle` with probability 0.9, the penalty is small; if it said 0.05,
the penalty is large.

The **optimizer** is **Adam**, a widely used variant of gradient descent that
adapts how large each step is. You do not need its internals; you need to know
it is the thing that actually changes the weights.

**Accuracy** is a separate, human-friendly number: the fraction of predictions
that are correct. Loss guides learning; accuracy tells you how it is going.

### 4.3 Underfitting, overfitting, and early stopping

- **Underfitting**: the model is too simple or has trained too little.
  Training accuracy is low. It has not learned the pattern.
- **Overfitting**: the model has begun memorizing individual training
  recordings, including their noise. Training accuracy keeps improving while
  validation accuracy stalls or gets worse. It has learned the examples rather
  than the gesture.

Overfitting is very easy with 45 recordings, so training uses **early
stopping**: it watches validation loss, and if 8 epochs pass with no
improvement, it stops and **restores the weights from the best epoch**, not the
last. That last part matters — without it you would keep the overfitted model
you just spent 8 epochs producing.

The reading guide for your curves:

| Pattern | Meaning | Response |
| --- | --- | --- |
| Both losses fall together | Healthy learning | Continue |
| Training loss falls, validation loss rises | Overfitting | Early stopping handles it; more or more varied data is the real fix |
| Both losses stay high and flat | Underfitting | Check data preparation; consider more units |
| Validation loss zig-zags | Small validation set (about 7 recordings) | Normal here; judge the trend, not one epoch |

---

## 5. Honest evaluation

### 5.1 Accuracy is not enough

Overall accuracy hides *where* the mistakes are. With three classes, a model
that is excellent at two of them and useless at the third can still post a
respectable-looking accuracy.

### 5.2 The confusion matrix

A **confusion matrix** is a table where **rows are the true labels** and
**columns are what the model predicted**. Correct answers land on the diagonal;
every off-diagonal cell is a specific type of mistake.

```
                    predicted
                circle  left_right  up_down
actual circle  [   3         0          0   ]
   left_right  [   0         2          1   ]
      up_down  [   0         1          2   ]
```

Read it: all `circle` examples were correct. One `left_right` was mistaken for
`up_down`, and one `up_down` was mistaken for `left_right`. That is an
actionable finding — those two sweeps are being confused with each other — and
it points you at your collection technique, not at the model.

### 5.3 Recall

**Recall** for a class is the fraction of that class's examples the model found:

```
recall = diagonal cell for the class ÷ total in that class's row
```

In the example above, `circle` has recall 3/3 = 1.00 while `left_right` has
2/3 = 0.67. Recall exposes a weak gesture that overall accuracy would hide.

### 5.4 The honesty rule

Look at the test set once, at the end. If you keep tweaking the model, checking
the test score, tweaking again, then you are slowly fitting the test set by
hand, and its number stops meaning "performance on unseen data". If one gesture
performs badly, the right response is usually to review or recollect *that
gesture's recordings*, not to keep re-rolling the model against the test score.

---

## Day 1 — Prepare examples

**Objective:** turn 45 CSV files into one array bundle with clean splits.

### Run it

```bash
python week2-model-training/prepare_data.py
```

### Expected output

```
Saved 45 recordings to week2-model-training/prepared_data.npz
- train: 30 {'circle': 10, 'left_right': 10, 'up_down': 10}
- validation: 6 {'circle': 2, 'left_right': 2, 'up_down': 2}
- test: 9 {'circle': 3, 'left_right': 3, 'up_down': 3}
```

Your exact numbers depend on how many recordings you collected. What matters:

- The total equals your number of CSV files.
- **Every split contains all three classes.** If a class is missing from a
  split, collect more recordings of it and rerun.

### What was created

`prepared_data.npz` is a compressed bundle of named arrays — like a small zip
file of tables. It contains, for each split, the inputs (`x_train`,
`x_validation`, `x_test`), the correct answers as numbers 0/1/2 (`y_train` and
so on), and the source filenames (`recording_train` and so on) so the validator
can later prove no recording crossed splits. It also stores the class names and
the six normalization means and standard deviations.

### Try it yourself 1

Run `prepare_data.py` twice and compare the printed counts. They are identical,
because seed 42 makes the shuffle reproducible. Reproducibility is what lets
you compare Monday's result with Thursday's and know the difference came from
your change and not from luck.

### Definition of done

- [ ] Week 1 validation passes first.
- [ ] Preparation reports train, validation, and test counts.
- [ ] Every split includes all three classes.
- [ ] You can explain why one recording cannot cross splits.

---

## Day 2 — Train

**Objective:** train the network and read its learning curves.

### Start the notebook

```bash
jupyter notebook week2-model-training/train.ipynb
```

A browser tab opens. A **notebook** is a document of alternating text cells and
runnable code cells; you run a cell with **Shift + Enter**, and cells share
state in order from top to bottom.

Choose **Run All** from the toolbar or the Cell/Run menu. Then go back and read
each markdown cell together with the code cell beneath it. Running everything
first gets the slow part started; reading afterwards is where the learning
happens.

### What each cell does

1. **Setup and load.** Fixes random seeds to 42 for Python, NumPy, and
   TensorFlow so training is reproducible, finds `prepared_data.npz` whether
   you launched from the repository root or from inside the week folder, and
   prints the array shapes. Confirm the shapes end in `(128, 6)`.
2. **Build the model.** The `Flatten → Dense(48, relu) → Dense(3, softmax)`
   stack from Section 4.1, compiled with the Adam optimizer and
   `sparse_categorical_crossentropy` loss. "Sparse" simply means the labels
   are integers (0, 1, 2) rather than one-hot vectors. `model.summary()` prints
   the parameter count — around 37 000.
3. **Train.** Up to 80 epochs with batch size 8 and early stopping with
   patience 8. Expect a few minutes. `verbose=2` prints one line per epoch:
   ```
   Epoch 12/80
   4/4 - 0s - loss: 0.2143 - accuracy: 0.9333 - val_loss: 0.3517 - val_accuracy: 0.8333
   ```
   `loss`/`accuracy` are on training data; `val_loss`/`val_accuracy` are on the
   validation set, which the model never learns from.
4. **Plot curves.** Two charts, loss and accuracy, training versus validation.
   Interpret them with the table in Section 4.3.

### Definition of done

- [ ] Every notebook cell finishes without an exception.
- [ ] Training and validation curves are visible.
- [ ] The best weights are restored by early stopping.
- [ ] You can say whether your curves show healthy learning, overfitting, or
      underfitting.

---

## Day 3 — Evaluate

**Objective:** measure the model on recordings it has never seen, and save the
evidence.

The last two notebook cells do this. Run them (they run as part of **Run All**)
and read the output carefully.

### What you get

```
Test accuracy: 88.9%
Per-class recall: {'circle': 1.0, 'left_right': 0.667, 'up_down': 1.0}
[[3 0 0]
 [0 2 1]
 [0 0 3]]
```

and three files in `models/`:

| File | Contents |
| --- | --- |
| `gesture_model.keras` | The trained model, weights and architecture |
| `metrics.json` | Accuracy, loss, confusion matrix, per-class recall, test filenames |
| `confusion_matrix.png` | The same matrix as a labeled image |

### How to read your own result

1. **Note the test set size.** With ~9 test recordings, each one is worth about
   11 percentage points. 88.9% versus 100% is one recording. Do not
   over-interpret small differences — this is the single most important
   statistical caution in the whole project.
2. **Inspect every row of the matrix**, not just the total.
3. **Find the largest off-diagonal cell.** That is your worst confusion, and
   it belongs in `NOTES.md`.
4. **If one gesture is clearly weak**, ask why physically. Were those
   recordings collected while you were tired? Is that gesture genuinely similar
   to another the way you perform it? Review or recollect before you touch the
   model.

### Write it down now

Open `NOTES.md` and fill in an entry: test accuracy, per-class recall, the
largest confusion, and your interpretation. Doing it now, while the numbers are
in front of you, is far easier than reconstructing them in Week 3.

### Definition of done

- [ ] Overall test accuracy is recorded in `NOTES.md`.
- [ ] Recall for every class is recorded.
- [ ] The largest off-diagonal confusion is identified.
- [ ] The test set was not used to choose epochs or model size.

---

## Day 4 — One controlled experiment

**Objective:** learn how to change a model responsibly.

### The method

A **controlled experiment** changes exactly one factor and holds everything
else fixed. If you change three things and the result improves, you have
learned nothing about which one helped. This is the core discipline of
practical machine learning, and it is what turns a project into engineering.

### The experiment

1. **Save the baseline first.** Copy your current test accuracy, per-class
   recall, and the size of `models/gesture_model.keras` into `NOTES.md`. Get
   the file size with:
   ```bash
   ls -l week2-model-training/models/gesture_model.keras     # macOS and Linux
   dir week2-model-training\models\gesture_model.keras       # Windows
   ```
2. **Change one number.** In the model-building cell, change:
   ```python
   tf.keras.layers.Dense(48, activation="relu"),
   ```
   to `Dense(24, ...)`.
3. **Rerun the whole notebook** (Run All), so the model is rebuilt and
   retrained from scratch rather than continuing from the old weights.
4. **Record the new accuracy and model size.**
5. **Decide and restore.** Keep whichever configuration you can justify. If the
   24-unit model is worse, set it back to 48 and **rerun the notebook**, so
   that the saved artifacts match the model you claim to have. The validators
   check the files, not your intentions.

### What you should observe

Halving the hidden layer roughly halves the parameter count (about 37 000 to
about 18 500), so the file shrinks noticeably. Accuracy usually changes only a
little, and with 9 test recordings that change may be pure noise. That is the
lesson: **smaller is often nearly free, and small accuracy differences on a
small test set are not evidence.**

### Try it yourself 2

Write two sentences in `NOTES.md` explaining the trade-off you measured: what
you gave up in accuracy, what you gained in size, and whether the difference is
larger than one test recording.

### Definition of done

- [ ] The baseline result was saved before changing anything.
- [ ] Exactly one factor changed.
- [ ] Both accuracy and model size were compared.
- [ ] The chosen final model was regenerated by rerunning the notebook.

---

## Day 5 — Prove completion

**Objective:** have an independent program confirm your results.

### Run the validator

```bash
python week2-model-training/validate_week2.py
```

### What it checks, and why

| Check | Why it exists |
| --- | --- |
| `x_*` arrays have shape `(*, 128, 6)` | Wrong shapes mean preparation went wrong |
| Inputs and labels have equal lengths | A mismatch means examples and answers are misaligned |
| No recording name appears in two splits | Direct proof of no data leakage |
| Training data is all finite | `nan` silently destroys training |
| `gesture_model.keras` exists and is > 1 KiB | Proof a real model was saved |
| `confusion_matrix.png` exists and is > 1 KiB | Proof the evaluation was actually rendered |
| `metrics.json` has a 3 × 3 matrix with entries | Proof examples were really evaluated |
| Per-class recall lists all three labels | Nothing was quietly dropped |
| `test_accuracy` ≥ 0.70 | Minimum evidence the pipeline works |

Note that the validator re-opens the files rather than trusting anything the
notebook printed. This is the same idea as a lab notebook: the artifact is the
evidence.

### Passing output

```
Week 2 validation PASSED: test accuracy 88.9%
- evaluated recordings: 9
- model size: 148.6 KiB
```

### Try it yourself 3

Rename `models/metrics.json` to `metrics.json.bak`, run the validator, read the
exact failure it reports, then rename it back. Learning to read a validator's
message is a skill you will use for the rest of the project.

### About the 70% threshold

70% is a floor that shows the pipeline works end to end — not a target, and not
a quality guarantee. With a small, single-person dataset, accuracy can swing
several points between runs. Report your exact number and its test-set size,
never "about 90%".

If you cannot reach 70%, work through this order:

1. Confirm Week 1 validation still passes.
2. Confirm every split has all three classes.
3. Look at the confusion matrix: is one gesture responsible? Recollect that
   gesture with more consistent technique.
4. Collect more recordings per gesture — more data beats model tweaking at this
   scale almost every time.
5. Only then try model changes, one at a time, recording each in `NOTES.md`.

Never edit the validator's threshold.

### Definition of done

- [ ] Validation reports `PASSED`.
- [ ] Test accuracy is at least 70%.
- [ ] All generated evidence comes from the final model.
- [ ] `PROGRESS.md` and `NOTES.md` are current.

---

## 11. Code walkthrough

### 11.1 `prepare_data.py`

```python
LABELS = ("circle", "left_right", "up_down")
FEATURES = ("ax", "ay", "az", "gx", "gy", "gz")
WINDOW_SIZE = 128
SEED = 42
```

The four decisions that the rest of the project depends on. `FEATURES` fixes
the column *order*; Week 3's live code must use the same order or predictions
become meaningless without any error appearing.

```python
def read_recording(path: Path) -> np.ndarray:
    ...
    if reader.fieldnames != ["timestamp_ms", *FEATURES]:
        raise ValueError(f"{path.name}: unexpected CSV header")
```

Reads by column *name*, and rejects an unexpected header rather than guessing.
`timestamp_ms` is intentionally not used as a feature: the model should learn
the shape of the motion, not the clock.

```python
if len(values) < 10 or not np.isfinite(values).all():
    raise ValueError(f"{path.name}: too few rows or non-finite data")
```

A second line of defence after Week 1's validator. Bad data should stop the
program loudly, not quietly become a bad model.

```python
def resample(values: np.ndarray) -> np.ndarray:
    old_positions = np.linspace(0.0, 1.0, num=len(values))
    new_positions = np.linspace(0.0, 1.0, num=WINDOW_SIZE)
    columns = [
        np.interp(new_positions, old_positions, values[:, index])
        for index in range(values.shape[1])
    ]
    return np.stack(columns, axis=1).astype(np.float32)
```

Section 3.1 in code. Both position arrays run 0 → 1, so a recording of any
length maps onto the same normalized timeline. Each of the six features is
interpolated **separately**, which is correct: they are independent
measurements. `np.stack(..., axis=1)` reassembles them into `(128, 6)`.

```python
rng = np.random.default_rng(SEED)
for label_index in range(len(LABELS)):
    indices = np.flatnonzero(labels == label_index)
    if len(indices) < 10:
        raise ValueError(f"{LABELS[label_index]} needs at least 10 recordings")
    rng.shuffle(indices)
    train_end = int(len(indices) * 0.70)
    validation_end = train_end + max(1, int(len(indices) * 0.15))
```

The stratified split: it loops over classes and splits each one separately, so
every split gets every class. `max(1, ...)` guarantees at least one validation
recording per class even with few files. Whole recordings are moved — the code
only ever indexes complete recordings.

```python
train_values = raw_windows[splits["train"]]
mean = train_values.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
std = train_values.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
std = np.maximum(std, np.float32(1e-6))
normalized = (raw_windows - mean) / std
```

The critical lines. Statistics come **only** from `splits["train"]`.
`axis=(0, 1)` averages over recordings and time but not features, producing six
means and six standard deviations. The accumulation runs in float64 for
precision. `np.maximum(std, 1e-6)` prevents division by zero if some feature
were perfectly constant. The saved statistics are then applied to *all* splits —
which is right: test data must be transformed the same way, using training
numbers.

```python
arrays[f"recording_{name}"] = np.asarray([paths[i].name for i in indices])
```

Filenames travel with the data so `validate_week2.py` can independently prove
no recording appears in two splits.

### 11.2 `train.ipynb`

```python
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
```

Three separate random number generators are involved — Python's, NumPy's, and
TensorFlow's — and all three must be seeded for a reproducible run.

```python
week_dir = Path.cwd()
if not (week_dir / "prepared_data.npz").exists():
    candidate = week_dir / "week2-model-training"
```

A small convenience: the notebook works whether Jupyter was started from the
repository root or from inside the week folder.

```python
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(128, 6)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(48, activation="relu"),
    tf.keras.layers.Dense(len(labels), activation="softmax"),
])
```

`Sequential` means "one layer after another". Declaring `Input(shape=(128, 6))`
makes the expected shape explicit, so a preparation mistake fails immediately
with a clear message instead of much later.

```python
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=8, restore_best_weights=True
)
```

Watches validation loss, tolerates 8 epochs without improvement, and — the part
that matters — restores the best weights rather than the final ones.

```python
matrix = np.zeros((len(labels), len(labels)), dtype=int)
for actual, predicted in zip(y_test, predictions):
    matrix[int(actual), int(predicted)] += 1
```

The confusion matrix is built with an explicit loop rather than a library
function, so you can see exactly what it counts: one increment per test
example, at [true row, predicted column].

```python
per_class_recall = {
    label: float(matrix[index, index] / matrix[index].sum())
    if matrix[index].sum() else 0.0
    for index, label in enumerate(labels)
}
```

Recall as defined in Section 5.3, with a guard against dividing by zero if a
class somehow has no test examples.

```python
model.save(models_dir / "gesture_model.keras")
(models_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
```

The model and its measurements are saved together, including the list of test
filenames, so a later reader can verify exactly which recordings produced the
score.

### 11.3 `validate_week2.py`

```python
overlap = names.intersection(str(item) for item in recordings)
if overlap:
    errors.append(f"recordings leak between splits: {sorted(overlap)}")
names.update(str(item) for item in recordings)
```

The leakage proof: it accumulates recording names across splits and reports any
name that shows up twice. This is why filenames were carried through
preparation.

```python
if not model_path.exists() or model_path.stat().st_size < 1000:
```

Existence *and* substance. A zero-byte file would otherwise pass a naive
existence check.

```python
if sum(sum(int(value) for value in row) for row in matrix) <= 0:
    errors.append("confusion_matrix does not contain evaluated examples")
```

An all-zero matrix would mean nothing was actually evaluated — a plausible
outcome of a broken cell, and one that would otherwise slip through.

---

## 12. Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `No CSV recordings found ... Complete Week 1 first.` | Wrong folder, or empty `data/` | Run from the repository root; confirm Week 1 validation passes |
| `circle needs at least 10 recordings` | Not enough files for that label | Collect more; the target is 15 per label |
| `unexpected CSV header` | A hand-edited or foreign CSV in `data/` | Remove it and re-record with `collect_data.py` |
| `ModuleNotFoundError: No module named 'tensorflow'` | Environment not activated, or package missing | Activate `.venv`, then `python -m pip install tensorflow` |
| `jupyter: command not found` | Jupyter not installed in this environment | `python -m pip install jupyter` |
| Notebook: `FileNotFoundError: prepared_data.npz` | `prepare_data.py` not run, or Jupyter started elsewhere | Run preparation; launch Jupyter from the repository root |
| TensorFlow prints many `oneDNN`/`cuDNN` notices | Informational messages, not errors | Ignore them |
| Training accuracy near 1.0, validation much lower | Overfitting on a small dataset | Expected; early stopping handles it. More data is the real fix |
| Validation curve zig-zags wildly | Only ~6 validation recordings | Normal; judge the trend |
| `test_accuracy is 0.556; need at least 0.70` | Data quality or class confusion | Follow the ordered checklist in Day 5 |
| `recordings leak between splits` | `prepared_data.npz` is stale or hand-modified | Delete it and rerun `prepare_data.py` |
| `models/confusion_matrix.png is missing or too small` | The plotting cell did not run | Run All in the notebook again |
| Results differ between two runs | A seed was changed, or cells ran out of order | Restart the kernel and Run All |

---

## 13. Self-check questions and answers

1. **Why must all inputs have shape `(128, 6)` when your recordings have
   different row counts?**
   A neural network's first layer has a fixed number of weights, so it needs a
   fixed input size. Interpolation resamples each recording onto the same
   128-point timeline.

2. **Why compute the normalization mean and standard deviation from training
   data only?**
   Using all the data would let information about the test set influence
   preparation. The test set must stay completely unseen for its score to mean
   anything.

3. **What is the difference between the validation set and the test set?**
   Validation is consulted *during* training to decide things like when to
   stop. Test is opened once at the very end. Using test data for decisions
   would gradually fit the model to it.

4. **Your model gets 100% on training data and 60% on validation data. What is
   happening?**
   Overfitting: it is memorizing individual recordings rather than learning the
   gestures. Early stopping limits the damage; more and more varied data is the
   real remedy.

5. **What does softmax do, and why is a large softmax value not proof of
   correctness?**
   It converts three raw scores into probabilities summing to 1. It always
   produces a confident-looking distribution over the three known classes, even
   for a movement that is none of them.

6. **Your accuracy is 89% and one gesture has recall 0.67. Which do you report?**
   Both, along with the test-set size. Overall accuracy alone hides the weak
   class, and 0.67 on three examples means two out of three.

7. **Why change only one thing in the Day 4 experiment?**
   So that any difference in the result can be attributed to that change.
   Changing several factors makes the outcome uninterpretable.

8. **Why does the validator reopen the saved files instead of trusting the
   notebook output?**
   Printed output can come from a stale cell or a model you have since changed.
   The saved artifacts are the actual evidence.

---

## 14. Glossary

| Term | Meaning |
| --- | --- |
| **Accuracy** | Fraction of predictions that are correct |
| **Activation function** | Non-linear function applied to a layer's output; ReLU and softmax here |
| **Adam** | The optimizer used; an adaptive form of gradient descent |
| **Backpropagation** | The calculus that determines how each weight should change |
| **Batch** | A small group of examples processed together; 8 here |
| **Bias** | A learned offset added inside a unit |
| **Class** | One possible answer; here `circle`, `left_right`, `up_down` |
| **Classification** | Choosing one answer from a fixed set |
| **Confusion matrix** | Table of true labels (rows) versus predictions (columns) |
| **Cross-entropy loss** | Penalty that grows as the correct class's probability falls |
| **Dense layer** | Layer where every input connects to every unit |
| **Early stopping** | Halting when validation loss stops improving, restoring the best weights |
| **Epoch** | One full pass over the training data |
| **Feature** | One input measurement; six per moment here |
| **Flatten** | Reshaping a grid into a single row without changing values |
| **Interpolation** | Estimating values between known points; used for resampling |
| **Loss function** | The number training tries to minimize |
| **Mean** | Average value |
| **Model** | A function whose internal numbers are learned from examples |
| **Normalization** | Rescaling features to comparable ranges using mean and standard deviation |
| **Optimizer** | The rule that updates weights from gradients |
| **Overfitting** | Memorizing training examples instead of learning the pattern |
| **Parameter / weight** | One learned number inside the model |
| **Recall** | Fraction of one class's examples that were found correctly |
| **ReLU** | `max(0, x)`; keeps positives, zeroes negatives |
| **Resampling** | Converting a recording to a fixed number of time steps |
| **Seed** | Fixed starting value making random choices reproducible |
| **Softmax** | Converts scores into probabilities summing to 1 |
| **Standard deviation** | Typical spread of values around the mean |
| **Stratified split** | Splitting each class separately so all classes appear everywhere |
| **Supervised learning** | Learning from labeled examples |
| **Tensor** | A rectangular block of numbers, described by its shape |
| **Test set** | Data used once, at the end, for an unbiased score |
| **Training set** | Data the model learns from |
| **Underfitting** | Failing to learn the pattern at all |
| **Validation set** | Data used during training to guide decisions |

---

## 15. Exit criteria

Week 2 is complete — and the project is 75% done — when all of these are true:

- [ ] All Day 1–5 definitions of done are checked.
- [ ] You can explain tensor shape, resampling, normalization, the three
      splits, epoch, loss, overfitting, and recall in your own words.
- [ ] `models/gesture_model.keras`, `models/metrics.json`, and
      `models/confusion_matrix.png` all come from the same final training run.
- [ ] `python week2-model-training/validate_week2.py` prints `PASSED` without
      any change to its requirements.
- [ ] `NOTES.md` contains your accuracy, per-class recall, largest confusion,
      and the Day 4 experiment.

Then continue to [Week 3](../week3-deployment/README.md), which shrinks this
model and runs it on live motion.
