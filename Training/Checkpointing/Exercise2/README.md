# Exercise 2 — Checkpointing in C

## Introduction

This exercise extends the checkpoint/restart concepts introduced in Exercise 1 to the C programming language.

As before, the example is a Monte Carlo estimate of $\pi$: random points $(x,y)$ are thrown into a unit square, and the fraction landing inside the unit circle ($x^2+y^2 \leq 1$) converges to $\pi/4$:

$$
\pi \approx 4 \times \frac{\text{points inside circle}}{\text{total points}}
$$

The same calculation is hardened through four progressively more robust versions:

1. `pi_naive.c` — no checkpointing
2. `pi_checkpoint.c` — periodic application-level checkpointing
3. `pi_signal.c` — signal-aware checkpointing
4. `pi_requeue.c` — automatic checkpoint/restart for the CANNON `serial_requeue` partition

The underlying workflow is the same as in the Python exercise:

```text
compute → checkpoint → compute → [interruption]
        → restart → load checkpoint → continue computing
```

The C examples expose some lower-level details that Python normally hides, including:

- explicit representation of application state
- binary checkpoint files
- random-number generator state
- `fflush()` and `fsync()`
- atomic `rename()`
- Unix signal handling with `sigaction()`
- `volatile sig_atomic_t` flags

---

## Content

| File | Description |
|---|---|
| `pi_naive.c` | Baseline Monte Carlo calculation with no checkpointing |
| `pi_checkpoint.c` | Adds periodic binary checkpointing and manual `--resume` |
| `pi_signal.c` | Adds safe handling of `SIGUSR1` and `SIGTERM` |
| `pi_signal.sbatch` | Slurm script requesting an early `SIGUSR1` warning |
| `pi_requeue.c` | Adds automatic checkpoint discovery and restart |
| `pi_requeue.sbatch` | Slurm script for the CANNON `serial_requeue` partition |
| `Makefile` | Builds all four C programs |

---

## Workflow

### 1. No checkpointing — `pi_naive.c`

The baseline program keeps all of its computational state only in process memory.

The important state consists of:

```text
completed darts
darts inside the circle
random-number generator state
```

If the process terminates, all unfinished work is lost.

Unlike the Python version, the C example uses a small deterministic `xorshift64*` pseudorandom-number generator rather than the standard `rand()` function.

Its complete internal state is represented by a single integer:

```c
uint64_t rng_state;
```

This makes it straightforward to save and restore the exact random sequence in the checkpointed versions.

---

### 2. Periodic checkpointing — `pi_checkpoint.c`

The second version introduces an explicit checkpoint structure:

```c
typedef struct {
    char magic[8];
    uint32_t version;

    uint64_t completed_darts;
    uint64_t inside_circle;
    uint64_t rng_state;

    uint64_t total_darts;
    uint64_t seed;

    double elapsed_seconds;
} checkpoint_t;
```

Every `--checkpoint-every` darts, this state is saved to a binary checkpoint file.

The checkpoint therefore contains everything required to continue the calculation:

```text
progress
numerical state
RNG state
run configuration
elapsed runtime
```

As in the Python example, checkpoints are written using a temporary-file/atomic-replacement pattern:

```text
write pi_checkpoint.bin.tmp
        ↓
      fflush()
        ↓
       fsync()
        ↓
      fclose()
        ↓
      rename()
        ↓
pi_checkpoint.bin
```

The existing checkpoint is not overwritten until the new checkpoint has been written successfully.

On restart, the `--resume` option loads the checkpoint and continues from the next dart.

Because `rng_state` is restored along with the counters, the resumed calculation continues the same pseudorandom-number sequence that an uninterrupted calculation would have produced.

> In this workshop example, the C structure is written directly to a binary file. This is convenient for demonstrating checkpointing, but direct structure serialization is not a fully portable production file format. Padding, byte order, data representation, and version compatibility should be considered in production applications.

---

### 3. Signal-aware checkpointing — `pi_signal.c`

The third version adds Unix signal handling.

Slurm can request that a signal be delivered shortly before a job reaches its wall-time limit:

```bash
#SBATCH --signal=USR1@60
```

The program installs handlers for:

```text
SIGUSR1
SIGTERM
```

The important C-specific pattern is that the signal handler performs no checkpoint I/O.

Instead, it only sets flags:

```c
static volatile sig_atomic_t stop_requested = 0;
static volatile sig_atomic_t received_signal = 0;

static void signal_handler(int signum)
{
    stop_requested = 1;
    received_signal = signum;
}
```

The normal program flow later checks the flag:

```text
Slurm / Unix
      |
      | signal
      v
signal handler
      |
      | set flag
      v
main computation
      |
      | reach safe point
      v
save checkpoint
      |
      v
exit cleanly
```

This avoids performing complex operations such as `printf()`, `fwrite()`, or memory allocation directly from the signal handler.

The main loop performs checkpoint I/O only after reaching a well-defined safe point between Monte Carlo iterations.

---

### 4. Automatic restart — `pi_requeue.c`

The final version is designed specifically for the CANNON `serial_requeue` partition.

Jobs running on `serial_requeue` may be preempted and requeued. Periodic checkpoints therefore act as the primary protection against losing completed work.

Unlike the previous checkpointed programs, `pi_requeue.c` does not require:

```text
--resume
```

At startup, it automatically checks whether a checkpoint exists:

```text
program starts
      |
      v
checkpoint exists?
     / \
   no   yes
   |     |
new run  load checkpoint
           |
           v
        continue
```

A Slurm requeue restarts the batch script from the beginning. When `pi_requeue.c` starts again, it finds the existing checkpoint and automatically resumes the computation.

The program also reports:

```text
SLURM_JOB_ID
SLURM_RESTART_COUNT
```

so that requeue events are visible in the job output.

Periodic checkpoints remain the safety mechanism even if a preemption occurs without enough advance warning to write a final checkpoint.

---

## Running

### Setup

Load the GNU compiler available on CANNON:

```bash
module load gcc
```

To inspect available versions:

```bash
module spider gcc
```

For reproducible builds, a specific GCC version can also be loaded explicitly.

Compile the programs before running them:

```bash
make
```

This should create:

```text
pi_naive
pi_checkpoint
pi_signal
pi_requeue
```

You can also compile an individual program manually. For example:

```bash
gcc \
    -std=c11 \
    -O2 \
    -Wall \
    -Wextra \
    -pedantic \
    -o pi_naive \
    pi_naive.c \
    -lm
```

Create a directory for checkpoint files:

```bash
mkdir -p checkpoints
```

The first two examples should be run interactively on a compute node:

```bash
salloc \
    --partition=test \
    --time=00:20:00 \
    --cpus-per-task=1 \
    --mem=1G
```

If needed after entering the allocation:

```bash
module load gcc
```

---

### 1. `pi_naive.c` — interactive

Run:

```bash
./pi_naive \
    --darts 50000000 \
    --seed 42 \
    --report-every 1000000 \
    --sleep 1
```

The `--sleep` option artificially slows the demonstration so that the program is easy to interrupt.

You should see output similar to:

```text
darts =      1000000 / 50000000   pi = ...
darts =      2000000 / 50000000   pi = ...
darts =      3000000 / 50000000   pi = ...
```

After several reports, press:

```text
Ctrl-C
```

Run the same command again.

The calculation begins again at dart 1 because no state was persisted.

---

### 2. `pi_checkpoint.c` — interactive

First remove any previous checkpoint:

```bash
rm -f \
    checkpoints/pi_checkpoint.bin \
    checkpoints/pi_checkpoint.bin.tmp
```

Start the calculation:

```bash
./pi_checkpoint \
    --darts 50000000 \
    --seed 42 \
    --report-every 1000000 \
    --checkpoint-every 5000000 \
    --checkpoint-file checkpoints/pi_checkpoint.bin \
    --sleep 1
```

Every five million darts, the program should report:

```text
--> checkpoint saved at dart 5000000
```

followed later by:

```text
--> checkpoint saved at dart 10000000
```

and so on.

Inspect the binary checkpoint:

```bash
ls -lh checkpoints/pi_checkpoint.bin
```

After at least two checkpoints have been written, press:

```text
Ctrl-C
```

Suppose the calculation is interrupted at approximately 13 million darts:

```text
0M ----- 5M ----- 10M ----- 13M
         CKPT      CKPT       X
                    ^
                    |
              last checkpoint
```

The work between 10 million and 13 million darts is lost, but the first 10 million darts are preserved.

Resume with the same parameters plus `--resume`:

```bash
./pi_checkpoint \
    --darts 50000000 \
    --seed 42 \
    --report-every 1000000 \
    --checkpoint-every 5000000 \
    --checkpoint-file checkpoints/pi_checkpoint.bin \
    --sleep 1 \
    --resume
```

You should see something similar to:

```text
========================================================================
Restarting from checkpoint
========================================================================
Completed darts  : 10000000
Inside circle    : ...
Previous runtime : ...
========================================================================
```

The program continues from the next dart instead of starting again from the beginning.

---

### 3. `pi_signal.c` — interactive and `sbatch`

#### Interactive signal test

Remove any previous checkpoint:

```bash
rm -f \
    checkpoints/pi_signal.bin \
    checkpoints/pi_signal.bin.tmp
```

Start the program in the background:

```bash
./pi_signal \
    --darts 100000000 \
    --seed 42 \
    --report-every 1000000 \
    --checkpoint-every 0 \
    --checkpoint-file checkpoints/pi_signal.bin \
    --sleep 1 &
```

Capture its process ID:

```bash
PID=$!
echo $PID
```

Allow the program to run briefly:

```bash
sleep 10
```

Then send `SIGUSR1`:

```bash
kill -USR1 $PID
```

Wait for the program to terminate:

```bash
wait $PID
```

Because:

```bash
--checkpoint-every 0
```

disables periodic checkpointing, the resulting checkpoint was produced specifically in response to the signal.

You should see output similar to:

```text
Signal received: SIGUSR1
Checkpointing at dart ...
Checkpoint saved: checkpoints/pi_signal.bin
Exiting cleanly.
```

Resume with:

```bash
./pi_signal \
    --darts 100000000 \
    --seed 42 \
    --report-every 1000000 \
    --checkpoint-every 0 \
    --checkpoint-file checkpoints/pi_signal.bin \
    --sleep 1 \
    --resume
```

---

#### Slurm signal test

The same mechanism can be demonstrated through `pi_signal.sbatch`:

```bash
#!/bin/bash

#SBATCH --job-name=pi-c-signal
#SBATCH --partition=test
#SBATCH --time=00:03:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

#SBATCH --output=pi_c_signal_%j.out
#SBATCH --error=pi_c_signal_%j.err

#SBATCH --signal=USR1@60

set -euo pipefail

module load gcc

mkdir -p checkpoints

echo "Job ID : ${SLURM_JOB_ID}"
echo "Node   : $(hostname)"
echo "Start  : $(date)"

srun ./pi_signal \
    --darts 10000000000 \
    --seed 42 \
    --report-every 1000000 \
    --checkpoint-every 0 \
    --checkpoint-file checkpoints/pi_signal.bin

echo "End: $(date)"
```

Remove any existing checkpoint:

```bash
rm -f \
    checkpoints/pi_signal.bin \
    checkpoints/pi_signal.bin.tmp
```

Submit:

```bash
sbatch pi_signal.sbatch
```

Monitor:

```bash
squeue -u $USER
```

Follow the output:

```bash
tail -f pi_c_signal_JOBID.out
```

Replace `JOBID` with the actual Slurm job ID.

The directive:

```bash
#SBATCH --signal=USR1@60
```

requests that Slurm send `SIGUSR1` as the job approaches its wall-time limit.

The application detects that signal at a safe point, saves a checkpoint, and exits cleanly.

This is a planned scheduler warning and should not be confused with unexpected preemption.

---

### 4. `pi_requeue.c` — `sbatch` only

> **This example is designed specifically for the CANNON `serial_requeue` partition.**

The batch script is:

```bash
#!/bin/bash

#SBATCH --job-name=pi-c-requeue
#SBATCH --partition=serial_requeue
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

#SBATCH --output=pi_c_requeue_%j.out
#SBATCH --error=pi_c_requeue_%j.err

#SBATCH --open-mode=append
#SBATCH --requeue

set -euo pipefail

module load gcc

mkdir -p checkpoints

echo "============================================================"
echo "Starting C requeue example"
echo "============================================================"
echo "Job ID        : ${SLURM_JOB_ID}"
echo "Restart count : ${SLURM_RESTART_COUNT:-0}"
echo "Node          : $(hostname)"
echo "Time          : $(date)"
echo "============================================================"

srun ./pi_requeue \
    --darts 1000000000 \
    --seed 42 \
    --report-every 1000000 \
    --checkpoint-every 5000000 \
    --checkpoint-file checkpoints/pi_requeue.bin

echo
echo "Batch script finished: $(date)"
```

Compile the program before submitting the batch job:

```bash
module load gcc
make
```

Do not compile inside the batch script.

Before the initial submission, remove any checkpoint from an earlier run:

```bash
rm -f \
    checkpoints/pi_requeue.bin \
    checkpoints/pi_requeue.bin.tmp
```

Submit and save the job ID:

```bash
JOBID=$(sbatch --parsable pi_requeue.sbatch)
echo $JOBID
```

Check its state:

```bash
squeue -j $JOBID
```

Follow the output:

```bash
tail -f pi_c_requeue_${JOBID}.out
```

Initially, the program should report:

```text
Slurm restart count : 0

No checkpoint found.
Starting a new calculation.
```

As the calculation progresses, periodic checkpoints appear:

```text
--> periodic checkpoint saved at dart 5000000
```

---

#### Simulate a requeue

Wait until at least one checkpoint has been written.

Then manually requeue the job:

```bash
scontrol requeue $JOBID
```

This provides a controlled way to demonstrate restart behavior without waiting for an actual `serial_requeue` preemption.

Check the job:

```bash
squeue -j $JOBID
```

When the job starts again, Slurm executes the batch script from the beginning.

Because:

```bash
#SBATCH --open-mode=append
```

is used, the existing output log is preserved and the new output is appended.

Continue following the same file:

```bash
tail -f pi_c_requeue_${JOBID}.out
```

After the restart you should see something similar to:

```text
Slurm job ID        : 12345678
Slurm restart count : 1

CHECKPOINT FOUND - AUTOMATIC RESTART
Completed darts  : 25000000
Inside circle    : ...
Previous runtime : ...
```

No `--resume` option is required.

The program automatically discovers the checkpoint and resumes.

---

#### What happens during a real `serial_requeue` interruption?

For example:

```text
0M --- 5M --- 10M --- 15M --- 20M --- 23M
       CKPT     CKPT     CKPT     CKPT      X
                                           |
                                      preemption
```

The computation between 20 million and 23 million darts may be lost.

However:

```text
checkpoints/pi_requeue.bin
```

contains the state at 20 million darts.

After Slurm requeues the job:

```text
preemption
     |
     v
job returns to queue
     |
     v
new allocation
     |
     v
batch script starts from beginning
     |
     v
./pi_requeue starts
     |
     v
checkpoint detected
     |
     v
state restored
     |
     v
continue computation
```

Periodic checkpoints are therefore the primary protection against unexpected preemption.

---

#### `SLURM_RESTART_COUNT`

The program reports the environment variable:

```bash
SLURM_RESTART_COUNT
```

The original execution normally shows:

```text
Slurm restart count : 0
```

After one requeue:

```text
Slurm restart count : 1
```

and after another:

```text
Slurm restart count : 2
```

This makes requeue events easy to identify in the application log.

---

#### Do not use `--fresh` inside the batch script

`pi_requeue` supports:

```bash
--fresh
```

to deliberately discard an old checkpoint and begin a new calculation.

For example:

```bash
./pi_requeue --fresh ...
```

However, **do not add `--fresh` to `pi_requeue.sbatch`**.

A requeued Slurm job starts the batch script again from the beginning. If `--fresh` were hardcoded into the script, every restart would discard the checkpoint that the job needs to recover from.

To start a genuinely new batch calculation, delete the checkpoint before the initial submission:

```bash
rm -f checkpoints/pi_requeue.bin
```

and then submit the job.

---

## Example output

### Resuming `pi_checkpoint`

```text
========================================================================
Restarting from checkpoint
========================================================================
Completed darts  : 10000000
Inside circle    : ...
Previous runtime : ...
========================================================================
```

---

### `pi_signal` after receiving `SIGUSR1`

```text
Signal received: SIGUSR1
Checkpointing at dart 12345678...
Checkpoint saved: checkpoints/pi_signal.bin
Exiting cleanly.
```

---

### `pi_requeue` after automatic restart

```text
========================================================================
Monte Carlo Pi - C Slurm Requeue Example
========================================================================
Slurm job ID        : 12345678
Slurm restart count : 1
Checkpoint file     : checkpoints/pi_requeue.bin
========================================================================

CHECKPOINT FOUND - AUTOMATIC RESTART
Completed darts  : 25000000
Inside circle    : ...
Previous runtime : ...
```

---

## Key concepts illustrated

- **Application state** — the minimum information needed to continue the calculation: progress, numerical state, RNG state, and configuration.

- **Explicit RNG state** — the `xorshift64*` generator has a single `uint64_t` state that can be saved and restored exactly.

- **Periodic checkpointing** — limits lost computation to approximately one checkpoint interval.

- **Binary checkpointing** — the application explicitly defines the structure of the saved state.

- **Checkpoint versioning** — the checkpoint includes a magic identifier and version number so incompatible files can be detected.

- **Atomic checkpoint replacement** — write a temporary file, `fflush()`, `fsync()`, close it, then `rename()` it over the previous checkpoint.

- **Signal-safe design** — a signal handler only modifies `volatile sig_atomic_t` flags; checkpoint I/O happens later in normal program execution.

- **Signal-triggered checkpointing** — an application can respond to scheduler warnings such as `SIGUSR1`.

- **Automatic restart** — `pi_requeue.c` automatically loads an existing checkpoint without requiring a manual `--resume` option.

- **Requeue-aware design** — Slurm restarts the batch workflow rather than restoring the previous process memory; the application must recover from persistent state.

- **Application-level vs. system-level checkpointing** — the C program explicitly chooses which state is required for restart. Later exercises will contrast this with transparent checkpoint/restart systems such as DMTCP.

---

## Python vs. C checkpointing

The Python and C exercises implement the same fundamental ideas using different mechanisms:

| Concept | Python | C |
|---|---|---|
| Application state | Python objects/dictionary | C structure |
| RNG state | `rng.getstate()` | `uint64_t rng_state` |
| Serialization | `pickle` | Binary `fwrite()` |
| Flush userspace buffer | `f.flush()` | `fflush()` |
| Flush to storage | `os.fsync()` | `fsync()` |
| Atomic replacement | `os.replace()` | `rename()` |
| Signal handling | `signal.signal()` | `sigaction()` |
| Signal flag | Python global | `volatile sig_atomic_t` |
| Resume | restore objects | restore structure fields |
| Requeue logic | automatic file detection | automatic file detection |

The implementation details differ, but the checkpoint/restart model is the same.

---

## Notes

- `pi_naive.c` and `pi_checkpoint.c` are intended to be run interactively so the interruption/restart cycle can be observed directly.

- `pi_signal.c` can be tested interactively with:

  ```bash
  kill -USR1 $PID
  ```

  or through Slurm's `--signal` mechanism using `pi_signal.sbatch`.

- `pi_requeue.c` is specifically designed for the CANNON `serial_requeue` partition.

- `scontrol requeue $JOBID` provides a convenient way to simulate requeue behavior during the workshop without waiting for a real preemption.

- Compile the C programs before submitting batch jobs rather than recompiling them inside each `sbatch` script.

- `module load gcc` loads the current default GCC module. For reproducibility, a specific GCC module version can be selected with:

  ```bash
  module spider gcc
  ```

- The binary checkpoint format used here is intentionally simple. Directly dumping a C structure is not guaranteed to be portable between different architectures, compilers, or future structure layouts. Production applications often use an explicit serialization format or libraries such as HDF5.

- Clean up generated checkpoint files and logs between experiments:

  ```bash
  rm -f checkpoints/pi_checkpoint.bin checkpoints/pi_checkpoint.bin.tmp
  rm -f checkpoints/pi_signal.bin checkpoints/pi_signal.bin.tmp
  rm -f checkpoints/pi_requeue.bin checkpoints/pi_requeue.bin.tmp

  rm -f pi_c_signal_*.out pi_c_signal_*.err
  rm -f pi_c_requeue_*.out pi_c_requeue_*.err
  ```

- Rebuild everything with:

  ```bash
  make
  ```

- Remove compiled executables with:

  ```bash
  make clean
  ```

The main lesson of this exercise is:

> **Checkpointing is an application design pattern, not a Python feature. In C, the same concepts become explicit: define the state, serialize it safely, persist it, and restore it after interruption.**

