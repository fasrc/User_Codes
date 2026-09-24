# Exercise 1 — Checkpointing in Python

## Introduction

This exercise introduces checkpoint/restart, the technique HPC applications use
to survive interruptions — job preemption, wall-time limits, node failures —
without losing all completed work.

The vehicle is a Monte Carlo estimate of $\pi$: random points $(x, y)$ are
thrown into a unit square, and the fraction landing inside the unit circle
($x^2 + y^2 \le 1$) converges to $\pi / 4$:

$$
\pi \approx 4 \times \frac{\text{points inside circle}}{\text{total points}}
$$

The same calculation is hardened through four progressively more robust
versions:

1. `pi_naive.py` — no checkpointing
2. `pi_checkpoint.py` — periodic application-level checkpointing
3. `pi_signal.py` — signal-aware checkpointing
4. `pi_requeue.py` — automatic checkpoint/restart for the CANNON `serial_requeue` partition

Every checkpointed version follows the same underlying pattern:

```
compute → checkpoint → compute → [interruption] → restart → load checkpoint → continue computing
```

---

## Content

| File | Description |
|------|-------------|
| `pi_naive.py` | Baseline Monte Carlo calculation, no checkpointing |
| `pi_checkpoint.py` | Adds periodic, atomic checkpointing with manual `--resume` |
| `pi_signal.py` | Adds a `SIGUSR1` handler that checkpoints on scheduler warning |
| `pi_signal.sbatch` | SLURM submission script requesting an early termination signal |
| `pi_requeue.py` | Adds automatic checkpoint discovery for SLURM requeue |
| `pi_requeue.sbatch` | SLURM submission script for the `serial_requeue` partition |

---

## Workflow

### 1. No checkpointing — `pi_naive.py`

Darts are thrown in a loop and progress (`completed_darts`, `inside_circle`,
RNG state) exists only in process memory. There is nothing to inspect or
recover once the process exits — an interruption at any point discards all
completed work.

### 2. Periodic checkpointing — `pi_checkpoint.py`

Every `--checkpoint-every` darts, the program serializes its state —
completed darts, darts inside the circle, RNG state, total darts, seed, and
elapsed runtime — to disk. The RNG state (`rng.getstate()` /
`rng.setstate(...)`) is saved alongside the counters so a resumed run
continues the exact same random sequence an uninterrupted run would have
produced, not just the same dart count. This matters even more for AI/ML
training, where reproducibility of the data/RNG stream is often required.

Checkpoints are written atomically to avoid a corrupted file if the process
dies mid-write:

```
write pi_checkpoint.pkl.tmp → flush/fsync → atomic rename → pi_checkpoint.pkl
```

If the write fails partway, the previous valid checkpoint is untouched. On
restart, `--resume` loads the checkpoint and continues from the next dart.

### 3. Signal-aware checkpointing — `pi_signal.py`

SLURM can warn a job that it is approaching its time limit by sending a Unix
signal ahead of termination (`#SBATCH --signal=USR1@60`). The program installs
a handler for `SIGUSR1` that only sets a flag — it performs no I/O inside the
handler itself:

```
SLURM/Unix → SIGUSR1 → handler sets flag → main loop reaches a safe point
    → save checkpoint → exit cleanly
```

The main loop checks that flag between iterations and, once set, saves a
checkpoint and exits cleanly instead of waiting to be killed by SLURM.

### 4. Automatic restart — `pi_requeue.py`

Jobs on the `serial_requeue` partition can be preempted and requeued by
SLURM at any time. `pi_requeue.py` combines periodic checkpointing with
*automatic* checkpoint discovery: unlike the previous examples, it does not
require `--resume`. On startup it checks for an existing checkpoint file and
loads it if present, so a requeued batch script — which SLURM always restarts
from the top — transparently picks up where the preempted run left off.

The environment variable `SLURM_RESTART_COUNT` (0 on the first run,
incrementing on each requeue) makes requeue events visible in application
logs.

---

## Running

### Setup

All examples use the Python module available on CANNON; no additional
packages are required:

```bash
module load python
```

The first two examples should run on a compute node rather than a login
node:

```bash
salloc --partition=test --time=00:20:00 --cpus-per-task=1 --mem=1G
module load python
mkdir -p checkpoints
```

### 1. `pi_naive.py` (interactive)

```bash
python pi_naive.py \
    --darts 50000000 \
    --seed 42 \
    --report-every 1000000 \
    --sleep 1
```

(`--sleep` artificially slows the loop so it's easy to interrupt.) Press
`Ctrl-C` after a few progress reports, then rerun the same command — the
calculation restarts from dart 1, since nothing was persisted.

### 2. `pi_checkpoint.py` (interactive)

```bash
rm -f checkpoints/pi_checkpoint.pkl checkpoints/pi_checkpoint.pkl.tmp

python pi_checkpoint.py \
    --darts 50000000 \
    --seed 42 \
    --report-every 1000000 \
    --checkpoint-every 5000000 \
    --checkpoint-file checkpoints/pi_checkpoint.pkl \
    --sleep 1
```

After at least two checkpoints have been written, press `Ctrl-C`, then
resume with the same parameters plus `--resume`:

```bash
python pi_checkpoint.py \
    --darts 50000000 \
    --seed 42 \
    --report-every 1000000 \
    --checkpoint-every 5000000 \
    --checkpoint-file checkpoints/pi_checkpoint.pkl \
    --sleep 1 \
    --resume
```

### 3. `pi_signal.py` (interactive and `sbatch`)

Interactive signal test — start the program in the background, send it
`SIGUSR1` after a few seconds, and wait for it to exit:

```bash
rm -f checkpoints/pi_signal.pkl checkpoints/pi_signal.pkl.tmp

python pi_signal.py \
    --darts 100000000 \
    --seed 42 \
    --report-every 1000000 \
    --checkpoint-every 0 \
    --checkpoint-file checkpoints/pi_signal.pkl \
    --sleep 1 &
PID=$!

sleep 10
kill -USR1 $PID
wait $PID
```

`--checkpoint-every 0` disables periodic checkpointing so the resulting
checkpoint is produced only by the signal. Resume it the same way as
before, with `--resume`.

On CANNON, the same mechanism runs under SLURM via `pi_signal.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=pi-signal
#SBATCH --partition=test
#SBATCH --time=00:03:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=pi_signal_%j.out
#SBATCH --error=pi_signal_%j.err
#SBATCH --signal=USR1@60

module load python

srun python pi_signal.py \
    --darts 10000000000 \
    --seed 42 \
    --report-every 1000000 \
    --checkpoint-every 0 \
    --checkpoint-file checkpoints/pi_signal.pkl
```

```bash
rm -f checkpoints/pi_signal.pkl checkpoints/pi_signal.pkl.tmp
sbatch pi_signal.sbatch
squeue -u $USER
tail -f pi_signal_JOBID.out   # replace JOBID with the actual job ID
```

`#SBATCH --signal=USR1@60` asks SLURM to deliver `SIGUSR1` 60 seconds before
the wall-time limit — a planned scheduler warning, not the same thing as
unexpected preemption.

### 4. `pi_requeue.py` (`sbatch` only)

Designed specifically for the CANNON `serial_requeue` partition. The
submission script:

```bash
#!/bin/bash
#SBATCH --job-name=pi-requeue
#SBATCH --partition=serial_requeue
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=pi_requeue_%j.out
#SBATCH --error=pi_requeue_%j.err
#SBATCH --open-mode=append
#SBATCH --requeue

module load python

echo "Job ID        : ${SLURM_JOB_ID}"
echo "Restart count : ${SLURM_RESTART_COUNT:-0}"

srun python pi_requeue.py \
    --darts 1000000000 \
    --seed 42 \
    --report-every 1000000 \
    --checkpoint-every 5000000 \
    --checkpoint-file checkpoints/pi_requeue.pkl
```

`--open-mode=append` preserves the output file across requeues instead of
truncating it, and `--requeue` allows SLURM to requeue the job on
preemption. Submit and follow it:

```bash
rm -f checkpoints/pi_requeue.pkl checkpoints/pi_requeue.pkl.tmp

JOBID=$(sbatch --parsable pi_requeue.sbatch)
squeue -j $JOBID
tail -f pi_requeue_${JOBID}.out
```

Once at least one checkpoint has been written, simulate a preemption:

```bash
scontrol requeue $JOBID
```

The running process is stopped and the job returns to the queue; when it
runs again, the batch script starts from the top, `pi_requeue.py` finds the
existing checkpoint, and computation resumes automatically — no `--resume`
flag needed.

> **Do not add `--fresh` to `pi_requeue.sbatch`.** SLURM restarts the batch
> script from the beginning on every requeue, so a hardcoded `--fresh` would
> discard the checkpoint the restarted job is supposed to recover from. To
> start a genuinely new run, delete the checkpoint *before the initial
> submission* instead: `rm -f checkpoints/pi_requeue.pkl`.

---

## Example output

Resuming `pi_checkpoint.py`:

```
Checkpoint found.
Resuming from dart : 10,000,000
Inside circle      : ...
Previous runtime   : ...
```

`pi_signal.py` after receiving `SIGUSR1`:

```
========================================================================
Termination signal received
========================================================================
Signal             : SIGUSR1
Completed darts    : ...
Saving checkpoint ...
Checkpoint saved   : checkpoints/pi_signal.pkl
Exiting cleanly.
========================================================================
```

`pi_requeue.py` after an automatic restart:

```
Slurm job ID        : 12345678
Slurm restart count : 1

========================================================================
CHECKPOINT FOUND - AUTOMATIC RESTART
========================================================================
Completed darts  : 25,000,000
Inside circle    : ...
Previous runtime : ...
========================================================================
```

---

## Key concepts illustrated

- **Application state** — the minimum information needed to continue a
  calculation: progress, numerical state, RNG state, configuration
- **Periodic checkpointing** — bounding lost work to roughly one checkpoint
  interval
- **Atomic checkpoint writes** — write-tmp / fsync / rename, so a crash
  mid-write never corrupts the last good checkpoint
- **RNG state persistence** — required for a resumed stochastic calculation
  to reproduce the same sequence an uninterrupted run would have produced
- **Signal-triggered checkpointing** — reacting to a scheduler warning
  (`SIGUSR1`) instead of periodic polling alone
- **Automatic restart** — detecting and loading an existing checkpoint on
  startup, with no manual flag required
- **Requeue-aware design** — a SLURM requeue restarts the batch workflow,
  not the application's in-memory state; recovery must come from persistent
  storage

---

## Notes

- `pi_naive.py` and `pi_checkpoint.py` are meant to be run interactively so
  the checkpoint/restart cycle can be observed directly.
- `pi_signal.py` can be tested interactively (via `kill -USR1`) or through
  SLURM's `--signal` mechanism, which is the realistic use case.
- `pi_requeue.py` targets `serial_requeue` specifically; `scontrol requeue
  $JOBID` simulates a preemption without waiting for a real one.
- Clean up generated checkpoints and logs between runs:

  ```bash
  rm -f checkpoints/pi_checkpoint.pkl checkpoints/pi_checkpoint.pkl.tmp
  rm -f checkpoints/pi_signal.pkl checkpoints/pi_signal.pkl.tmp
  rm -f checkpoints/pi_requeue.pkl checkpoints/pi_requeue.pkl.tmp
  rm -f pi_signal_*.out pi_signal_*.err pi_requeue_*.out pi_requeue_*.err
  ```
