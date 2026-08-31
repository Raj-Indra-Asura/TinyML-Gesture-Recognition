# Week 2: Model Training

## Goals

This week you will transform recordings into consistent examples, train a small
neural network, and evaluate it on movements the model never practiced on. The
result is a saved model plus numerical and visual evidence of its quality.

## Concepts

### Inputs, classes, and tensors

A machine-learning **model** is a function whose internal numbers are learned
from examples. The six sensor columns are **features**: measurements the model
uses as clues. The three gestures are **classes**, the possible answers.

Libraries store model data in **tensors**, rectangular collections of numbers.
One input here has shape `(128, 6)`: 128 moments and six features. A batch of 10
inputs has shape `(10, 128, 6)`.

Recordings differ slightly in length because USB and sensor timing are not
perfect. **Interpolation** estimates values at 128 evenly spaced positions so
every example has the same shape. It does not invent another gesture.

### Normalization and splitting

Acceleration and rotation use different units and numeric ranges.
**Normalization** subtracts each training feature's mean and divides by its
standard deviation, a measure of typical spread. Features then have comparable
scales. Only training data determines those numbers; using test data would leak
information from the exam.

The **training set** adjusts the model. The **validation set** guides choices
during training. The **test set** is opened once for the final unbiased score.
This repository splits whole recordings, never rows, so one performance cannot
occur in two sets.

### Neural network training

A **dense layer** combines every incoming number using learned **weights**, adds
learned offsets, and applies an **activation**. ReLU (rectified linear unit)
keeps positive values and replaces negative values with zero. The final
**softmax** activation turns three scores into probabilities that add to one.

During an **epoch**, the model practices on every training example once.
**Cross-entropy loss** is a penalty that grows when the correct class receives
low probability. The Adam **optimizer** changes weights to reduce that loss.
Accuracy is the fraction of answers that are correct.

Too little learning is **underfitting**. Memorizing training examples while
validation performance worsens is **overfitting**. Early stopping restores the
weights from the epoch with the best validation loss.

### Honest evaluation

A **confusion matrix** has actual classes as rows and predicted classes as
columns. Its diagonal contains correct answers. **Recall** for one class is the
fraction of its test examples found correctly. It can expose a weak gesture
even when overall accuracy looks acceptable.

## Hands-on

### Day 1 — Prepare examples

Activate the environment from Week 1, then run:

```bash
python week2-model-training/prepare_data.py
```

Read the printed counts. Each split must contain every class. The command
creates `prepared_data.npz`, a compressed bundle of arrays, recording names,
labels, and normalization values.

**Try it yourself 1:** run preparation twice and compare the printed counts.
They remain identical because seed 42 makes the random split reproducible.

**Definition of done**

- [ ] Week 1 validation passes first.
- [ ] Preparation reports train, validation, and test counts.
- [ ] Every split includes all three classes.
- [ ] You can explain why one recording cannot cross splits.

### Day 2 — Train

Start Jupyter from the repository root:

```bash
jupyter notebook week2-model-training/train.ipynb
```

Choose **Run All**. Read each markdown cell before its code cell. Training may
take several minutes. The history chart should show falling loss; validation
loss may be noisy because the physical dataset is intentionally small.

**Definition of done**

- [ ] Every notebook cell finishes without an exception.
- [ ] Training and validation curves are visible.
- [ ] The best weights are restored by early stopping.

### Day 3 — Evaluate

The notebook evaluates the untouched test recordings, prints accuracy and
per-class recall, and saves the model, `metrics.json`, and a confusion matrix
image in `models/`. Inspect every matrix row. If one gesture is poor, first
review or recollect that gesture rather than repeatedly tuning against the test
set.

**Definition of done**

- [ ] Overall test accuracy is recorded in `NOTES.md`.
- [ ] Recall for every class is recorded.
- [ ] The largest off-diagonal confusion is identified.
- [ ] The test set was not used to choose epochs or model size.

### Day 4 — One controlled experiment

Copy your first metrics into `NOTES.md`. Change only the first dense layer from
48 to 24 units, rerun the training notebook, and compare model size and test
accuracy. Restore 48 units and rerun if the smaller model is worse. Changing one
factor at a time makes cause and effect interpretable.

**Try it yourself 2:** make the controlled 48-versus-24 comparison and explain
the trade-off in two sentences.

**Definition of done**

- [ ] The baseline result was saved before changing anything.
- [ ] Exactly one factor changed.
- [ ] Both accuracy and model size were compared.
- [ ] The chosen final model was regenerated.

### Day 5 — Prove completion

```bash
python week2-model-training/validate_week2.py
```

This command checks tensor shapes, finite values, recording leakage, a real
confusion matrix, per-class metrics, saved-model size, and at least 70% test
accuracy. A failure is a diagnosis, not permission to edit the validator.

**Try it yourself 3:** rename `metrics.json`, observe the validator's specific
failure, and restore the filename.

**Definition of done**

- [ ] Validation reports `PASSED`.
- [ ] Test accuracy is at least 70%.
- [ ] All generated evidence comes from the final model.
- [ ] `PROGRESS.md` and `NOTES.md` are current.

## Code walkthrough

`prepare_data.py` reads each complete CSV, identifies its label from the
filename, and interpolates each feature separately to 128 rows. It shuffles
recording indices with a fixed random generator and splits each class, which
keeps all groups represented. Mean and standard deviation come only from
training windows. The `.npz` bundle retains recording names so validation can
prove there is no overlap.

The notebook fixes random seeds, loads prepared arrays, and flattens each
`128 × 6` input before two dense layers. Flattening changes shape, not values.
Early stopping limits overfitting. The final cells predict the test set,
calculate the confusion matrix without another library, render it, and write
machine-readable metrics before saving the Keras model.

`validate_week2.py` independently opens these artifacts. It does not trust a
printed notebook result: it checks shapes, split membership, file substance,
metric ranges, and the number of actually evaluated examples.

## Validation

The passing threshold demonstrates a functioning pipeline, not universal
quality. With a small single-person dataset, accuracy can vary sharply. Preserve
the exact result and failure cases rather than presenting 70% as a guarantee.

## Exit criteria

- [ ] All Day 1–5 definitions of done are checked.
- [ ] You can explain tensors, normalization, splits, epoch, loss, and recall.
- [ ] The final Keras model and matching evaluation artifacts exist.
- [ ] The Week 2 validator passes without changing its requirements.
