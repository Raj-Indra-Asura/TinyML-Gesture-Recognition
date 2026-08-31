# Week 1: Data Foundations

## Goals

By Friday you will stream motion measurements from a board, collect at least 45
labeled gesture recordings, and prove that the files are usable. This week
matters most: a model cannot repair unclear or incorrectly labeled examples.

## Concepts

### The board, sensor, and USB connection

A **microcontroller** is a small computer made to control hardware. Your Arduino
contains an **inertial measurement unit (IMU)**: a sensor that measures motion.
Its accelerometer reports acceleration on the x, y, and z axes in units of
Earth's gravity, while its gyroscope reports rotation speed around those axes
in degrees per second.

The IMU talks to the microcontroller over **I2C** ("I squared C"), a short-range
two-wire connection between chips. The Arduino library handles those electrical
messages. The sketch then sends text to the laptop through **serial**, an
ordered stream of bytes carried by USB. A **baud rate** describes the serial
communication speed; both ends use 115200.

### Samples, axes, and frequency

One **sample** is the six values measured at one moment: `ax, ay, az, gx, gy,
gz`. The sketch waits 20 milliseconds between samples, giving a **sampling
frequency** of about 50 samples per second. Fifty samples per second preserves
ordinary hand motion without producing excessive data.

A comma-separated values (**CSV**) file stores one sample per row. Its header
names each column. `timestamp_ms` says how many milliseconds passed after a
recording began.

### Labels and trustworthy data

A **label** is the correct answer attached to an example. This course uses
`circle`, `left_right`, and `up_down`. A recording is one three-second
repetition, not a collection of unrelated movements. **Balanced data** means
each label has roughly the same number of recordings, preventing the model from
winning by favoring the most common gesture.

Keep repetitions separate. Later, all samples from one recording stay in the
same training, validation, or test group. This prevents **data leakage**, where
nearly identical pieces of one performance appear both in practice data and in
the final exam.

## Hands-on

### Day 1 — Stream measurements

1. Place the unplugged board on a dry, clear desk. Hold it by the edges; do not
   bend it or touch exposed pins with metal.
2. Connect it with a data-capable USB cable.
3. In Arduino IDE 2, install the `Arduino_BMI270_BMM150` library using Library
   Manager. Select **Arduino Nano 33 BLE** as the board and select its port.
4. Open `arduino/stream_imu/stream_imu.ino`, upload it, and open Serial Monitor
   at 115200 baud.
5. Confirm the header is followed by six comma-separated numbers. Carefully
   tilt the board; several values should change.
6. Close Serial Monitor before using Python. Only one program can own the port.

**Definition of done**

- [ ] The correct board and port are selected.
- [ ] Upload finishes without an error.
- [ ] Every data row contains six numbers.
- [ ] Tilting and rotating change different columns.

### Day 2 — Make a practice recording

Find the port in Arduino IDE or your operating system. From the repository root:

```bash
python week1-data-foundations/collect_data.py --port /dev/ttyACM0 --label circle
```

Replace `/dev/ttyACM0` with your port; Windows ports look like `COM4`. The
collector waits for the board to reconnect, prints `GO`, records three seconds,
and gives the output path.

Open the CSV in a text editor. Do not "clean" individual values by hand. If a
recording is wrong, delete the entire file and repeat it.

**Try it yourself 1:** collect once with `--seconds 5`. Compare its row count
with a three-second file, then delete the five-second practice file so the real
dataset stays consistent.

**Definition of done**

- [ ] Serial Monitor is closed while Python records.
- [ ] A CSV has the seven expected columns.
- [ ] A three-second recording has roughly 150 rows.
- [ ] You know how to delete and repeat a bad take.

### Day 3 — Physical collection

Use a seated or standing position you can reproduce. Clear an arm's length of
space. Keep the USB cable loose and route it away from your hand. Stop if the
board, cable, or connector becomes warm or damaged.

For every take:

1. Hold the board flat, component side up, between thumb and fingers. Use the
   same orientation throughout this dataset.
2. Rest in a comfortable position. Start the command with the correct label.
3. Wait for `GO`; begin within half a second.
4. Perform the gesture smoothly for the full three seconds:
   - **circle:** draw repeated circles about 20 cm wide, facing forward.
   - **left_right:** sweep horizontally about 30 cm, reversing smoothly.
   - **up_down:** sweep vertically about 30 cm, reversing smoothly.
5. Avoid sudden stops, cable tugs, extra wrist flicks, talking with your hands,
   or mixing another gesture into the take.
6. Rest for several seconds and check the saved row count before repeating.

Collect five rounds. In each round, record three repetitions of each gesture in
a changing order. This produces 15 independent recordings per label and reduces
the effect of fatigue or gradual position changes. Take a break between rounds.
If another person will use the final system, collect the same balanced set from
them only with their consent.

**Definition of done**

- [ ] At least 15 separate CSV files exist for every label.
- [ ] All gestures used the same board orientation and duration.
- [ ] Mistakes were repeated rather than relabeled.
- [ ] Movement was safe and the cable did not restrict it.

### Day 4 — Inspect quality

Read several files from the beginning, middle, and end of collection. Check that
timestamps increase, rows have the same width, and values vary during motion.
Compare filenames with your collection order. Large differences can be real;
do not remove an example merely because it looks unusual. Remove it only when
you know the label or collection procedure was wrong.

**Try it yourself 2:** change `SAMPLE_INTERVAL_US` in the sketch from `20000`
to `40000`, upload, and observe that a practice recording has about half as many
rows. Restore `20000` and upload again before continuing.

**Definition of done**

- [ ] At least one file per label was inspected.
- [ ] No known mislabeled or interrupted take remains.
- [ ] Every class still has at least 15 recordings.

### Day 5 — Prove completion

Run:

```bash
python week1-data-foundations/validate_week1.py
```

The validator reads every value, checks increasing timestamps, requires
100–250 rows per take, and requires 15 files for each label. Fix the named
recording or collect a replacement until it prints `PASSED`.

**Try it yourself 3:** temporarily move one CSV outside `data/`, run validation,
read the exact failure, and restore the file.

**Definition of done**

- [ ] Validation reports `PASSED`.
- [ ] It reports at least 45 recordings total.
- [ ] `PROGRESS.md` is updated.
- [ ] A lesson or failure is recorded in `NOTES.md`.

## Code walkthrough

The Arduino sketch starts USB serial and `IMU.begin()` before sampling. `micros`
supplies a clock; subtracting the previous time avoids blocking other work.
Availability checks prevent stale reads. Each `Serial.print` emits one field,
and commas make a valid six-column row.

`collect_data.py` validates command-line choices, opens serial with the same
baud rate, and discards startup text or broken rows. `time.monotonic()` is a
clock that never jumps when wall time changes. Samples remain in memory until
at least ten valid rows arrive, so a failed connection does not leave a
misleading CSV. UTC timestamps give every recording a unique, sortable name.

`validate_week1.py` derives labels from filenames, parses every field as a
finite number, checks row counts and timestamps, and reports all actionable
problems together.

## Validation

Passing `validate_week1.py` is necessary, but your physical checklist is also
part of the proof because software cannot see an incorrect hand movement.

## Exit criteria

- [ ] All Day 1–5 definitions of done are checked.
- [ ] There are balanced, physically collected examples of three gestures.
- [ ] You can explain IMU, sample, frequency, serial, label, and leakage.
- [ ] The Week 1 validator passes without editing its thresholds.
