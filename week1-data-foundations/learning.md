# Week 1: Data Foundations

**Project progress this week: 0% → 35%.**
Start here after reading [`README.md`](README.md).

This file is the only resource you need for Week 1. It contains the setup
instructions, the explanation of every idea, the exact commands, the expected
output, what to do when something goes wrong, and a line-by-line walkthrough of
all the code you will run. Read a section before running its commands.

---

## Table of contents

1. [What you are building and why this week matters](#1-what-you-are-building-and-why-this-week-matters)
2. [Set up your tools](#2-set-up-your-tools)
3. [Concepts you need before touching the board](#3-concepts-you-need-before-touching-the-board)
4. [Day 1 — Stream measurements](#day-1--stream-measurements)
5. [Day 2 — Make a practice recording](#day-2--make-a-practice-recording)
6. [Day 3 — Physical collection](#day-3--physical-collection)
7. [Day 4 — Inspect quality](#day-4--inspect-quality)
8. [Day 5 — Prove completion](#day-5--prove-completion)
9. [Code walkthrough](#9-code-walkthrough)
10. [Troubleshooting](#10-troubleshooting)
11. [Self-check questions and answers](#11-self-check-questions-and-answers)
12. [Glossary](#12-glossary)
13. [Exit criteria](#13-exit-criteria)

---

## 1. What you are building and why this week matters

### The finished system, in one paragraph

You hold a small circuit board in your hand and move it in a circle. The board
measures its own motion 50 times per second, a tiny neural network looks at the
last 128 measurements, and the words `circle, confidence=94.2%` appear on a
screen. That is the whole project. It is called **TinyML**: machine learning
that runs on a device with kilobytes of memory instead of a server.

### Week 1's share of the work

| Week | What it delivers | Cumulative |
| --- | --- | --- |
| **1 (this week)** | Working hardware link and a validated dataset of 45+ labeled recordings | **35%** |
| 2 | Prepared arrays, a trained model, a measured accuracy | 75% |
| 3 | A quantized model, live inference, a demo, honest limitations | 100% |

### Why data comes first

Machine learning does not "figure out" what you meant. It copies patterns from
the examples you give it. If ten of your `circle` recordings actually contain a
sloppy `left_right` sweep, the model learns that `circle` sometimes looks like
`left_right`, and no amount of clever training in Week 2 can undo it.

There is a saying for this: *garbage in, garbage out*. Week 1 is where you stop
the garbage from getting in. Collecting a clean recording takes ten seconds.
Discovering a dirty dataset in Week 3 costs you the whole project.

### Goals for this week

By Friday you will:

- Stream six live motion values from the board to your laptop.
- Collect at least 15 recordings of each of the three gestures (45 total).
- Prove with a script that every file is complete, correctly formed, and
  correctly labeled.

---

## 2. Set up your tools

Do this section once. If you already followed the setup in the root
[`README.md`](../README.md), skim it and move on.

### 2.1 Hardware you need

- **Arduino Nano 33 BLE Sense Rev2.** A small board with a processor and
  several sensors, including the motion sensor this project uses.
- **A data-capable USB cable.** This trips up more beginners than anything
  else. Some cheap USB cables carry power only. If your board's power light
  comes on but the Arduino IDE never lists a port for it, try a different
  cable before you try anything else.

### 2.2 Install the Arduino IDE

1. Download Arduino IDE 2 from `arduino.cc` and install it.
2. Open it. The large text area is where sketches (Arduino programs) live.
3. Install support for your board: **Tools → Board → Boards Manager**, search
   for `Arduino Mbed OS Nano Boards`, and click **Install**. This can take
   several minutes; it is downloading a compiler for the board's processor.
4. Install the sensor library: **Tools → Manage Libraries** (or the book icon
   in the left sidebar), search for `Arduino_BMI270_BMM150`, and click
   **Install**. `BMI270` and `BMM150` are the part numbers of the motion and
   magnetic sensor chips on the Rev2 board.

> A **library** is code someone else wrote that you can call. Without it you
> would have to write the low-level instructions that ask the sensor chip for
> its latest reading. With it, you write `IMU.readAcceleration(x, y, z)`.

### 2.3 Install Python and the packages

Check what you have. Open a terminal (Terminal on macOS/Linux, PowerShell on
Windows) and run:

```bash
python --version
```

You need 3.10 or newer. If the command is not found, try `python3 --version`;
if that also fails, install Python from `python.org` and, on Windows, tick
**Add Python to PATH** in the installer.

Now move into the project folder and create a **virtual environment** — a
private folder of Python packages for this project so it cannot break other
projects on your computer:

```bash
cd TinyML-Gesture-Recognition
python -m venv .venv
```

Activate it. You must do this in every new terminal you open for this project:

```bash
source .venv/bin/activate          # macOS and Linux
.venv\Scripts\Activate.ps1         # Windows PowerShell
```

Your prompt now starts with `(.venv)`. Install the packages:

```bash
python -m pip install pyserial numpy matplotlib tensorflow jupyter
```

Week 1 only needs `pyserial`, but installing everything now avoids
interruptions later.

| Package | What it does for you |
| --- | --- |
| `pyserial` | Lets Python read the USB serial stream from the board |
| `numpy` | Fast arrays and maths (Week 2) |
| `matplotlib` | Charts (Week 2) |
| `tensorflow` | Builds, trains, and converts the model (Weeks 2–3) |
| `jupyter` | Runs the training notebook (Week 2) |

### 2.4 Where to run commands

Every command in this course is run **from the repository root** — the folder
containing `README.md`, `week1-data-foundations/`, and so on. That is why
commands look like `python week1-data-foundations/collect_data.py` rather than
`python collect_data.py`. If you get `can't open file ... No such file or
directory`, you are almost certainly in the wrong folder. Run `pwd` (macOS and
Linux) or `cd` with no arguments (Windows) to see where you are.

---

## 3. Concepts you need before touching the board

Read all of Section 3 before Day 1. Every term in **bold** is also in the
[glossary](#12-glossary).

### 3.1 The board, the sensor, and the connection

A **microcontroller** is a small computer built to control hardware rather than
run applications. It has no operating system, no screen, and no hard disk. It
runs one program, forever, starting the moment it gets power.

Your board contains an **inertial measurement unit (IMU)**: a sensor that
measures its own motion. It has two parts you will use.

- The **accelerometer** measures acceleration along three perpendicular
  directions called the **x, y, and z axes**. The unit is **g**, one unit of
  Earth's gravity. Important and initially surprising: a board lying still on
  a desk does *not* read zero. It reads about `1.0` on whichever axis points
  up, because gravity is an acceleration. That means the accelerometer tells
  you both how the board is tilted and how it is being moved.
- The **gyroscope** measures how fast the board is *rotating* around each of
  those axes, in **degrees per second**. Lying still, it reads approximately
  zero on all three. Twist the board and one value spikes.

Together they give six numbers: `ax, ay, az` (acceleration) and `gx, gy, gz`
(rotation rate). These six numbers are everything the project knows about your
hand. It cannot see; it can only feel.

The sensor chip talks to the microcontroller over **I2C** (say
"eye-squared-see"), a two-wire electrical connection used between chips on the
same board. The library handles that entirely; you never write I2C code.

The microcontroller then talks to your laptop over **serial**: an ordered
stream of bytes travelling down the USB cable, one after another, like cars in
a single-file tunnel. Both ends must agree on the speed, called the **baud
rate**. This project uses **115200** everywhere. If the two ends disagree you
get an unreadable jumble of symbols rather than an error message.

**Only one program can hold a serial port at a time.** If the Arduino IDE's
Serial Monitor is open, Python cannot read the port, and vice versa. This
single fact explains most beginner failures in Week 1 and Week 3.

### 3.2 Samples, axes, and sampling frequency

One **sample** is the six values captured at one instant: `ax, ay, az, gx, gy,
gz`.

The sketch waits 20 000 microseconds — 20 milliseconds — between samples.
Because there are 1000 milliseconds in a second:

```
1000 ms per second ÷ 20 ms per sample = 50 samples per second
```

50 samples per second is the **sampling frequency**, written **50 Hz** (hertz
means "per second").

Why 50? Two competing pressures:

- **Too slow** and you miss the movement. A fast hand sweep that takes 0.3
  seconds would be described by only 3 samples at 10 Hz — not enough shape to
  tell one gesture from another.
- **Too fast** and you produce far more data than the movement contains, which
  costs memory and battery on a device that has very little of either.

Ordinary hand motion contains detail up to roughly 10–20 Hz, and 50 Hz captures
that comfortably. It is a deliberate engineering choice, not a default.

### 3.3 CSV files and timestamps

A **CSV** (comma-separated values) file is a plain text table. The first line
is the **header**, naming the columns; every line after it is one row of data:

```csv
timestamp_ms,ax,ay,az,gx,gy,gz
0.000000,0.012000,-0.004000,0.998000,0.310000,-0.120000,0.050000
20.031000,0.015000,-0.002000,0.997000,0.290000,-0.140000,0.060000
```

You can open a CSV in any text editor, and also in a spreadsheet program.

`timestamp_ms` is how many milliseconds have passed since that recording
started. It is written by the *laptop*, not the board. It always increases, and
consecutive values are about 20 ms apart, though not exactly — the operating
system, USB, and Python each add small unpredictable delays. Week 2 deals with
that unevenness deliberately.

### 3.4 Labels, balance, and leakage

A **label** is the correct answer attached to an example. This course uses
exactly three: `circle`, `left_right`, and `up_down`. The label lives in the
*filename*, for example `circle_20240101T101500Z.csv`. That is why the
collector refuses labels it does not recognise, and why you must never rename a
file to fix a bad take.

One **recording** is one repetition of one gesture, lasting three seconds. It
is not a mixture, and not "whatever I did for a while".

**Balanced data** means every label has roughly the same number of recordings.
Imagine you collected 40 `circle` files and 5 of each other gesture. A model
that ignores the sensor entirely and always answers `circle` would be right 80%
of the time. That number looks like success and means nothing. Balance removes
that cheap shortcut. This is why the validator demands 15 files per label.

**Data leakage** is when information from the final exam sneaks into the
practice material. In Week 2 your recordings get split into a training set, a
validation set, and a test set. If pieces of the *same* three-second
performance ended up in both training and test, the model would be scored on
something it has effectively already seen, and its reported accuracy would be
too optimistic. The defence starts here in Week 1: **keep every repetition in
its own file**, so that Week 2 can split whole files and never split rows.

### 3.5 Safety and consent

- Hold the board by its edges. Do not touch the exposed metal pins with rings,
  keys, or other metal.
- Keep it away from water and away from a strong pull on the cable.
- Give yourself an arm's length of clear space; look before you sweep.
- Stop immediately if the board, cable, or connector becomes hot or damaged.
- The recordings describe how *you* move. If you record another person, ask
  first and explain what the data is for. Do not commit anyone's raw
  recordings to Git — `.gitignore` already prevents this.

---

## Day 1 — Stream measurements

**Objective:** see six live numbers change when you move the board.

### Steps

1. Place the **unplugged** board on a dry, clear desk. Hold it by the edges.
2. Connect it to your laptop with a data-capable USB cable.
3. In Arduino IDE 2, choose **Tools → Board → Arduino Mbed OS Nano Boards →
   Arduino Nano 33 BLE**. (The Rev2 board uses this same entry.)
4. Choose **Tools → Port** and select the port that appeared when you plugged
   the board in. It looks like `/dev/ttyACM0` on Linux, `/dev/cu.usbmodem1101`
   on macOS, or `COM4` on Windows. **Write this port down — you need it for
   every Python command this week.**
5. Open the sketch: **File → Open**, then
   `week1-data-foundations/arduino/stream_imu/stream_imu.ino`.
6. Click **Upload** (the right-pointing arrow). The IDE compiles the sketch and
   copies it to the board. The first upload is slow.
7. Open **Tools → Serial Monitor**. In the dropdown at the right of the monitor
   panel, select **115200 baud**.

### What correct output looks like

```
ax,ay,az,gx,gy,gz
0.011963,-0.003906,0.998047,0.305176,-0.122070,0.061035
0.013428,-0.004395,0.997559,0.305176,-0.183105,0.061035
```

New lines should appear continuously, roughly 50 per second.

Now test it physically:

- **Tilt the board slowly.** The three acceleration values should shift, with
  roughly `1.0` moving between them depending on which way is up.
- **Twist it quickly and stop.** One gyroscope value should spike and return to
  near zero.
- **Hold it perfectly still.** The values should be nearly constant, with tiny
  fluctuations. That flicker is **sensor noise**, and it is normal.

If you understand *why* `az ≈ 1.0` when the board is flat, you understand the
accelerometer. It is feeling gravity.

8. **Close Serial Monitor** before you use Python. Only one program can own the
   port.

### Definition of done

- [ ] The correct board and port are selected.
- [ ] Upload finishes without an error.
- [ ] Every data row contains six numbers.
- [ ] Tilting and rotating change different columns.
- [ ] You wrote your port name down.

---

## Day 2 — Make a practice recording

**Objective:** save one gesture to a CSV file and understand what it contains.

### Steps

With Serial Monitor **closed** and your virtual environment activated, run this
from the repository root:

```bash
python week1-data-foundations/collect_data.py --port /dev/ttyACM0 --label circle
```

Replace `/dev/ttyACM0` with the port you wrote down on Day 1.

### What the arguments mean

| Argument | Required? | Meaning |
| --- | --- | --- |
| `--port` | yes | The serial port of your board |
| `--label` | yes | One of `circle`, `left_right`, `up_down`; anything else is rejected |
| `--seconds` | no | Recording length; default `3.0` |
| `--baud` | no | Serial speed; default `115200`, matching the sketch |
| `--output-dir` | no | Where CSVs are written; default `week1-data-foundations/data` |

### What you will see

```
Opening /dev/ttyACM0. Keep the board still while the connection starts...
GO: perform 'circle' smoothly for 3 seconds.
Saved 148 rows to week1-data-foundations/data/circle_20240101T101500Z.csv
```

The two-second pause before `GO` is deliberate: opening a serial port resets
the Arduino, and the board needs a moment to restart and begin sending. Keep
still during it, then start moving as soon as `GO` appears.

About 148 rows for three seconds is right: 3 seconds × 50 Hz = 150, minus a few
dropped or malformed lines. The validator accepts 100–250 rows.

### Inspect the file

Open the saved CSV in a text editor and check:

- The first line is exactly `timestamp_ms,ax,ay,az,gx,gy,gz`.
- Every row has seven values.
- `timestamp_ms` starts near 0 and increases by roughly 20 each row.
- The sensor values change from row to row while you were moving.

**Never hand-edit values inside a CSV.** Editing measurements is fabricating
data. If a recording is wrong, delete the whole file and record it again.

### Try it yourself 1

Collect once with a longer duration:

```bash
python week1-data-foundations/collect_data.py --port /dev/ttyACM0 --label circle --seconds 5
```

Compare its row count with the three-second file. You should see roughly 250
versus 150 — because rows arrive at a fixed 50 per second, so time and row
count are proportional. Then **delete the five-second file**, so that every
recording in your real dataset has the same duration.

### Definition of done

- [ ] Serial Monitor is closed while Python records.
- [ ] A CSV has the seven expected columns.
- [ ] A three-second recording has roughly 150 rows.
- [ ] You know how to delete and repeat a bad take.
- [ ] Both practice files are deleted before Day 3 begins.

---

## Day 3 — Physical collection

**Objective:** collect 15 clean recordings of each gesture — the core of the
project's 35%.

### Set up your body and your space

Use a seated or standing position you can reproduce every time. Clear an arm's
length of space. Route the USB cable so it hangs loose and never tugs your
hand mid-gesture. If the cable restricts you, your gestures will change shape
and your dataset will describe the cable, not the movement.

### Grip and orientation

Hold the board flat, component side up, between your thumb and fingers. **Use
the same orientation for every single recording, all week.** The IMU reports
motion relative to its own body. Turning the board over swaps the sign of some
axes, and a model trained on one orientation will fail on the other. This is
the most common silent mistake in the whole project.

### The three gestures

| Label | Movement | Size |
| --- | --- | --- |
| `circle` | Draw repeated circles in the air, facing forward, continuing for the full three seconds | about 20 cm across |
| `left_right` | Sweep horizontally left and right, reversing smoothly at each end | about 30 cm each way |
| `up_down` | Sweep vertically up and down, reversing smoothly at each end | about 30 cm each way |

All three are *repeated* for the whole three seconds. One slow circle followed
by stillness is a bad `circle` example.

### Procedure for every take

1. Rest in your starting position, board held correctly.
2. Run the command with the correct label:
   ```bash
   python week1-data-foundations/collect_data.py --port /dev/ttyACM0 --label left_right
   ```
3. Stay still until `GO` appears, then begin within about half a second.
4. Perform the gesture smoothly and continuously for the full three seconds.
5. Avoid: sudden stops, cable tugs, extra wrist flicks, gesturing while
   talking, and drifting into a different gesture halfway through.
6. Check the printed row count, rest a few seconds, and repeat.

### Collect in rounds, not in blocks

Do **five rounds**. In each round, record three repetitions of each gesture,
and vary the order between rounds.

That produces 5 × 3 = 15 recordings per label, 45 in total. Rounds matter
because of **drift**: as you get tired, or shift in your chair, or gradually
speed up, your movements change. If you recorded all 15 `circle` files first
and all 15 `up_down` files last, the model could learn "early-session style
versus late-session style" instead of learning the gestures. Interleaving
spreads that drift evenly across all three labels.

Take a real break between rounds. Rushing produces sloppy gestures, and sloppy
gestures are permanent — they live in the dataset for the rest of the project.

If someone else will use the finished system, collect the same balanced set
from them, with their consent.

### Definition of done

- [ ] At least 15 separate CSV files exist for every label.
- [ ] All gestures used the same board orientation and duration.
- [ ] Mistakes were re-recorded, never relabeled.
- [ ] Movement was safe and the cable did not restrict it.

---

## Day 4 — Inspect quality

**Objective:** find real problems without inventing new ones.

### What to check

Open several files — some from your first round, some from the middle, some
from your last round — and confirm:

1. The header is correct and every row has seven values.
2. `timestamp_ms` increases from top to bottom, with no repeats.
3. Sensor values genuinely vary; a file of nearly identical rows means you did
   not move, or you started late.
4. The filename's label matches what you actually performed at that time. Your
   collection order is your only record, so check it against the timestamps in
   the filenames.

### What *not* to do

Do not delete a recording because it "looks unusual". Real human movement
varies, and that variation is exactly what makes a model robust. Remove a
recording only when you know something was procedurally wrong: you performed
the wrong gesture, the cable snagged, you started three seconds late, or the
label is incorrect.

And when you do remove one, **collect a replacement**, or you will drop below
15 files for that label and fall out of balance.

### Try it yourself 2 — see the sampling frequency change

This experiment makes 50 Hz concrete rather than abstract.

1. In `arduino/stream_imu/stream_imu.ino`, change:
   ```cpp
   const unsigned long SAMPLE_INTERVAL_US = 20000;  // 50 samples per second.
   ```
   to `40000`.
2. Upload, then take a *practice* recording (not part of your dataset).
3. It will contain about 75 rows instead of 150, because 40 ms between samples
   is 25 samples per second.
4. **Restore `20000`, upload again, and delete the practice file.** Every file
   in your dataset must share the same sampling frequency, or Week 2 will be
   comparing movements measured at two different rates.

### Definition of done

- [ ] At least one file per label was inspected.
- [ ] No known mislabeled or interrupted take remains.
- [ ] Every class still has at least 15 recordings.
- [ ] The sketch is back at `SAMPLE_INTERVAL_US = 20000`.

---

## Day 5 — Prove completion

**Objective:** replace "I think it's fine" with a machine-checked `PASSED`.

### Run the validator

```bash
python week1-data-foundations/validate_week1.py
```

### What it checks

| Check | Rule | Reason |
| --- | --- | --- |
| Filename | Must start with `circle_`, `left_right_`, or `up_down_` | The label lives in the filename |
| Header | Exactly `timestamp_ms,ax,ay,az,gx,gy,gz` | Week 2 reads columns by name |
| Row width | Exactly 7 values per row | A short row means a truncated sample |
| Numbers | Every value parses as a finite number | `nan` or text would poison training |
| Timestamps | Strictly increasing | Out-of-order rows mean the stream was disturbed |
| Row count | Between 100 and 250 | Catches takes that were far too short or long |
| Balance | At least 15 files per label | Prevents a lazy majority-class model |

### Passing output

```
Week 1 validation PASSED: 45 valid recordings
- circle: 15
- left_right: 15
- up_down: 15
```

### Failing output

```
Week 1 validation FAILED:
- circle_20240101T101500Z.csv: 62 rows; expected 100–250
- up_down: found 14 recordings; need 15
```

Each line names a file and a specific problem. Fix the cause — delete and
re-record that take, or collect the missing one — and run the validator again.

**Do not edit the validator's thresholds.** Lowering a requirement does not
make your dataset better; it only removes the evidence that it is not.

### Try it yourself 3

Temporarily move one CSV outside `data/`, run the validator, read the exact
failure message it produces, then move the file back and confirm `PASSED`
returns. Knowing what a failure looks like before you cause one accidentally is
worth the two minutes.

### Definition of done

- [ ] Validation reports `PASSED`.
- [ ] It reports at least 45 recordings total.
- [ ] `PROGRESS.md` Week 1 boxes are ticked.
- [ ] A lesson or failure is recorded in `NOTES.md`.

---

## 9. Code walkthrough

You do not need to write any code this week, but you should be able to explain
every file you run. Open each file beside this section.

### 9.1 `arduino/stream_imu/stream_imu.ino`

An Arduino sketch has two required functions. `setup()` runs once when the board
powers on or resets. `loop()` runs over and over, forever, as fast as it can.

```cpp
const unsigned long SAMPLE_INTERVAL_US = 20000;  // 50 samples per second.
unsigned long previousSampleUs = 0;
```

The interval is in **microseconds** (millionths of a second), so 20 000 µs =
20 ms = 50 Hz. `previousSampleUs` remembers when the last sample was taken.

```cpp
Serial.begin(115200);
while (!Serial && millis() < 5000) {
}
```

Start serial at 115200 baud and wait — for at most five seconds — for the
laptop to open the connection. The time limit matters: without it, a board
running on a battery with nothing listening would hang forever.

```cpp
if (!IMU.begin()) {
  Serial.println("ERROR: IMU initialization failed");
  while (true) {
  }
}
```

Start the sensor. If it fails, print a message and deliberately stop in an
infinite loop. Stopping loudly is better than streaming meaningless numbers.

```cpp
Serial.println("ax,ay,az,gx,gy,gz");
```

Print the header once, so a human watching Serial Monitor knows the column
order.

```cpp
const unsigned long now = micros();
if (now - previousSampleUs < SAMPLE_INTERVAL_US) {
  return;
}
previousSampleUs = now;
```

This is the **non-blocking timer** pattern, and it is worth understanding.
`micros()` returns how long the board has been running. If not enough time has
passed, `loop()` simply returns and gets called again immediately. The
alternative, `delay(20)`, would freeze the entire program for 20 ms and prevent
it from doing anything else — fine here, a bad habit later.

```cpp
if (!IMU.accelerationAvailable() || !IMU.gyroscopeAvailable()) {
  return;
}
```

Ask whether a *fresh* reading exists. Without this you could read the same
stale value twice and record motion that never happened.

```cpp
Serial.print(ax, 6);
Serial.print(',');
...
Serial.println(gz, 6);
```

Print each value with 6 decimal places, separated by commas, and end the line
after the last one. `print` stays on the line; `println` ends it. Together they
produce exactly one valid six-column row per sample.

### 9.2 `collect_data.py`

```python
LABELS = ("circle", "left_right", "up_down")
HEADER = ("timestamp_ms", "ax", "ay", "az", "gx", "gy", "gz")
```

The three valid labels and the exact output header, written once and reused, so
the collector and the validator cannot drift apart.

```python
parser.add_argument("--label", required=True, choices=LABELS)
```

`choices` makes Python reject a typo like `--label circel` immediately, with a
clear message, before the board is even opened. Catching an error early is
better than discovering a mislabeled file in Week 2.

```python
def parse_sensor_row(line: str) -> tuple[float, ...] | None:
    parts = line.strip().split(",")
    if len(parts) != 6:
        return None
    ...
    return values if all(math.isfinite(value) for value in values) else None
```

Every incoming line is checked three ways: it must split into exactly six
pieces, each piece must parse as a number, and each number must be finite (not
infinity, not `nan`). Anything else returns `None` and is counted as malformed
rather than saved. This is what silently discards the header line and any
half-line that arrived while the connection was starting.

```python
with serial.Serial(args.port, args.baud, timeout=1) as board:
    time.sleep(2)
    board.reset_input_buffer()
```

Opening the port resets the board, so sleep two seconds while it restarts, then
throw away anything already buffered. Otherwise your recording would begin with
stale bytes from before `GO`.

```python
started = time.monotonic()
while time.monotonic() - started < args.seconds:
```

`time.monotonic()` is a clock that only ever moves forward. Ordinary wall-clock
time can jump — daylight saving, a network time correction — and a jump
mid-recording would corrupt the timestamps. This is a small, deliberate
correctness choice.

```python
if len(rows) < 10:
    print(f"Only received {len(rows)} valid rows; no file was saved.")
    return 1
```

Rows are held in memory and only written at the end. If the connection failed,
you get a clear message and **no file**, rather than a nearly empty CSV that
looks real and quietly damages your dataset later.

```python
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
destination = args.output_dir / f"{args.label}_{stamp}.csv"
```

The filename carries the label and a UTC timestamp. UTC is the same everywhere
in the world, so names never collide and always sort in collection order.

### 9.3 `validate_week1.py`

```python
matching_labels = [label for label in LABELS if path.name.startswith(f"{label}_")]
if len(matching_labels) != 1:
```

The label is recovered from the filename, and exactly one label must match.
A file the validator cannot classify is an error, not something to ignore.

```python
header = tuple(next(reader, ()))
if header != EXPECTED_HEADER:
    return [f"{path.name}: expected header ..."]
```

If the header is wrong, checking the rows is pointless, so it reports that one
problem and stops on this file.

```python
if values[0] <= previous_time:
    errors.append(f"{path.name}:{line_number}: timestamps do not increase")
```

Timestamps must strictly increase. Equal or decreasing values mean rows arrived
out of order or were duplicated.

```python
if len(errors) >= 10:
    errors.append(f"{path.name}: stopped after 10 errors")
    break
```

A thoroughly broken file would otherwise produce hundreds of lines of output.
Ten is enough to diagnose it.

```python
for label in LABELS:
    if counts[label] < MIN_RECORDINGS_PER_LABEL:
```

The balance check. Note that all errors are collected and printed together, so
one run tells you everything to fix instead of one problem at a time.

---

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| No port appears in Arduino IDE | Charge-only USB cable | Try a different cable first; then a different USB port |
| No port appears, cable is fine | Board not in a responsive state | Double-tap the small reset button; the orange LED should pulse, then upload again |
| `Could not read /dev/ttyACM0: ... Permission denied` (Linux) | Your user is not in the `dialout` group | `sudo usermod -a -G dialout $USER`, then log out and back in |
| `could not open port ... Access is denied` (Windows) | Serial Monitor or another program holds the port | Close Serial Monitor and any other terminal running a script |
| `Missing pyserial. Run: python -m pip install pyserial` | Virtual environment not activated, or package missing | Activate `.venv`, then `python -m pip install pyserial` |
| `Only received 0 valid rows; no file was saved.` | Wrong port, Serial Monitor open, or sketch not uploaded | Verify the port, close Serial Monitor, re-upload `stream_imu.ino` |
| Serial Monitor shows random symbols | Baud rate mismatch | Set the monitor to 115200 |
| Serial Monitor shows `ERROR: IMU initialization failed` | Wrong or missing sensor library | Install `Arduino_BMI270_BMM150` and re-upload |
| Recording has ~75 rows instead of ~150 | `SAMPLE_INTERVAL_US` was left at 40000 | Restore 20000, re-upload, re-record |
| `python: can't open file ... No such file or directory` | Wrong working directory | `cd` to the repository root |
| Many "Ignored N startup or malformed lines" | A few are normal; large numbers suggest cable or connection trouble | Reseat the cable; re-record the take |
| Validator: `62 rows; expected 100–250` | You started late or the connection dropped | Delete that file and record it again |
| Validator: `found 14 recordings; need 15` | You deleted a bad take without replacing it | Collect one more of that gesture |

---

## 11. Self-check questions and answers

Answer these out loud before moving to Week 2. Being able to explain your
project is part of the deliverable.

1. **The board is lying flat and still, but `az` reads about 1.0. Is the sensor
   broken?**
   No. The accelerometer measures acceleration including gravity, and gravity
   is about 1 g downward. A flat, still board correctly reports ~1.0 on the
   upward axis.

2. **Why 50 samples per second rather than 5 or 500?**
   5 Hz would blur a fast gesture into a handful of points; 500 Hz would
   produce ten times the data with no extra information about hand movement.
   50 Hz captures ordinary human motion and stays small enough for a
   microcontroller.

3. **Why must Serial Monitor be closed while `collect_data.py` runs?**
   A serial port can only be held by one program at a time. If the IDE holds
   it, Python receives nothing.

4. **You performed `up_down` but ran the command with `--label circle`. Why not
   just rename the file?**
   You should re-record instead. Renaming is fine in principle, but relying on
   memory to correct labels is exactly how mislabeled data enters a dataset.
   Delete and repeat: it costs ten seconds and leaves no doubt.

5. **What is data leakage, and what does Week 1 do about it?**
   It is the same information appearing in both training and test data,
   producing an inflated accuracy. Week 1 prevents it by keeping every
   repetition in its own file, so Week 2 can split whole recordings.

6. **Why does the collector wait two seconds before printing `GO`?**
   Opening the serial port resets the board. The pause lets it restart and
   begin streaming so the first rows are real data.

7. **Why not lower `MIN_RECORDINGS_PER_LABEL` to 10 to finish faster?**
   The validator is your evidence that the dataset is adequate. Weakening the
   check does not improve the data; it only hides the weakness until Week 2.

---

## 12. Glossary

| Term | Meaning |
| --- | --- |
| **Accelerometer** | Sensor measuring acceleration on x, y, z, in units of gravity (g); includes gravity itself |
| **Balanced data** | Every label has roughly the same number of examples |
| **Baud rate** | Serial communication speed; 115200 throughout this project |
| **CSV** | Comma-separated values: a plain-text table with a header line |
| **Data leakage** | Test information contaminating training data, producing falsely high scores |
| **Gyroscope** | Sensor measuring rotation speed around x, y, z, in degrees per second |
| **Hz (hertz)** | Times per second |
| **I2C** | Two-wire electrical connection between chips on one board |
| **IMU** | Inertial measurement unit: accelerometer plus gyroscope |
| **Label** | The correct answer attached to an example; here, in the filename |
| **Library** | Reusable code you call instead of writing yourself |
| **Microcontroller** | Small computer that runs one program directly on hardware |
| **Recording / take** | One three-second repetition of one gesture, stored in one file |
| **Sample** | The six values captured at one instant |
| **Sampling frequency** | Samples per second; 50 Hz here |
| **Sensor noise** | Small random fluctuation in readings even when nothing moves |
| **Serial** | An ordered byte stream over USB between board and laptop |
| **Sketch** | An Arduino program, with `setup()` and `loop()` |
| **TinyML** | Machine learning that runs on very small, low-power devices |
| **Virtual environment** | A per-project folder of Python packages (`.venv`) |

---

## 13. Exit criteria

Week 1 is complete — and the project is 35% done — when all of these are true:

- [ ] All Day 1–5 definitions of done are checked.
- [ ] There are balanced, physically collected examples of all three gestures.
- [ ] You can explain IMU, sample, sampling frequency, serial, label, balance,
      and leakage in your own words.
- [ ] `python week1-data-foundations/validate_week1.py` prints `PASSED` without
      any edit to its thresholds.
- [ ] `PROGRESS.md` and `NOTES.md` are up to date.

Then continue to [Week 2](../week2-model-training/README.md), which turns these
recordings into a trained model.
