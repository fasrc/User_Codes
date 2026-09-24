# Exercise 1 — Checkpointing in Python

## Introduction

Checkpoint/restart lets a long-running job survive interruptions — preemption,
wall-time limits, node failures — without losing completed work. The pattern is:

```
compute → checkpoint → compute → [interruption] → restart → load checkpoint → continue
```

The vehicle is a Monte Carlo estimate of $\pi$: random points $(x, y)$ are
thrown into a unit square, and the fraction landing inside the unit circle
($x^2 + y^2 \le 1$) converges to $\pi / 4$:

$$
\pi \approx 4 \times \frac{\text{points inside circle}}{\text{total points}}
$$

The same calculation is hardened in four steps: no checkpointing, periodic
checkpointing, signal-aware checkpointing, and automatic restart on the CANNON
`serial_requeue` partition. The first two run **interactively**, the last two
in **batch** mode.

Checkpointing is done at the application level: the program periodically
serializes its state to a file with Python's `pickle`, and on restart loads it
and continues from the next dart. The checkpoint holds the number of completed
darts, the number inside the circle, the random-number generator state
(`rng.getstate()`), the total darts and seed (validated on resume), and the
elapsed runtime.

Key ideas along the way:

- **Application state** — the minimum needed to continue: progress counters,
  numerical state, RNG state, configuration.
- **RNG state persistence** — a resumed run reproduces the exact random
  sequence of an uninterrupted run.
- **Atomic writes** — write to a temporary file, `fsync`, then rename, so a
  crash mid-write never corrupts the last good checkpoint.

---

## Content

| File | Description | Mode |
|------|-------------|------|
| `pi_naive.py` | Baseline, no checkpointing | Interactive |
| `pi_checkpoint.py` | Periodic, atomic checkpoints; manual `--resume` | Interactive |
| `pi_signal.py` | Adds a `SIGUSR1` handler that checkpoints on scheduler warning | Batch |
| `pi_signal.sbatch` | Starts a new `pi_signal.py` run with `--signal=USR1@60` | Batch |
| `pi_signal_restart.sbatch` | Restarts `pi_signal.py` from its checkpoint (`--resume`) | Batch |
| `pi_signal_15362.out` | Example output of the first `pi_signal.sbatch` run (stopped by `SIGUSR1` at 342,615,676 darts) | Batch |
| `pi_signal_restart_15363.out` | Example output of the `pi_signal_restart.sbatch` run (resumed from that checkpoint, stopped by `SIGUSR1` at 751,484,956 darts) | Batch |
| `pi_requeue.py` | Finds an existing checkpoint automatically on startup | Batch |
| `pi_requeue.sbatch` | Runs `pi_requeue.py` with `--requeue`; traps `SIGUSR1` (`--signal=B:USR1@60`) and requeues itself | Batch |
| `pi_requeue_15378.out` | Example output of a `pi_requeue.sbatch` run that was requeued twice (restarted from checkpoints at 370,686,091 and 705,774,009 darts) and then completed | Batch |

---

## Workflow

### Setup

Steps 1 and 2 run on a compute node, not a login node:

```bash
salloc --partition=test --time=00:20:00 --cpus-per-task=1 --mem=1G
module load python
mkdir -p checkpoints
```

No additional Python packages are needed. In all examples `--sleep 1` slows the
loop so there is time to interrupt it.

### 1. No checkpointing — `pi_naive.py` (interactive)

Progress lives only in process memory, so an interruption loses everything.

1. Start the calculation:

   ```bash
   python pi_naive.py \
       --darts 50000000 \
       --seed 42 \
       --report-every 1000000 \
       --sleep 1
   ```

   ```
   ...
   darts =    1,000,000 / 50,000,000   pi = 3.14024400   error = 1.349e-03   elapsed = 0.22 s
   darts =    2,000,000 / 50,000,000   pi = 3.14123400   error = 3.587e-04   elapsed = 1.44 s
   darts =    3,000,000 / 50,000,000   pi = 3.14173733   error = 1.447e-04   elapsed = 2.67 s
   darts =    4,000,000 / 50,000,000   pi = 3.14225500   error = 6.623e-04   elapsed = 3.89 s
   darts =    5,000,000 / 50,000,000   pi = 3.14223440   error = 6.417e-04   elapsed = 5.12 s
   ...
   ```

2. Press `Ctrl-C` after a few progress reports.
3. Rerun the same command. It starts again from dart 1 — nothing was saved.

### 2. Periodic checkpointing — `pi_checkpoint.py` (interactive)

Every `--checkpoint-every` darts the program saves the completed darts, darts
inside the circle, RNG state, run parameters and elapsed time. On restart,
`--resume` loads them and continues with the next dart.

1. Remove any old checkpoint:

   ```bash
   rm -f checkpoints/pi_checkpoint.pkl checkpoints/pi_checkpoint.pkl.tmp
   ```

2. Start the calculation:

   ```bash
   python pi_checkpoint.py \
       --darts 50000000 \
       --seed 42 \
       --report-every 1000000 \
       --checkpoint-every 5000000 \
       --checkpoint-file checkpoints/pi_checkpoint.pkl \
       --sleep 1
   ```

   ```
   ...
   darts =    5,000,000 / 50,000,000   pi = 3.14223440   error = 6.417e-04   elapsed = 7.18 s
     --> checkpoint saved at dart 5,000,000
   darts =    6,000,000 / 50,000,000   pi = 3.14191867   error = 3.260e-04   elapsed = 8.82 s
   ...
   ```

3. After at least two checkpoints (10,000,000 darts), press `Ctrl-C`.
4. Resume with the same parameters plus `--resume`:

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

   ```
   Checkpoint found.
   Resuming from dart : 10,000,000
   Inside circle      : 7,855,434
   Previous runtime   : 12.81 s

   darts =   11,000,000 / 50,000,000   pi = 3.14213600   error = 5.433e-04   elapsed = 13.09 s
   ...
   ```

   The run continues from the checkpoint instead of dart 1. The checkpoint must
   match `--darts` and `--seed`, otherwise the program refuses to resume.

### 3. Signal-aware checkpointing — `pi_signal.py` (batch)

SLURM can warn a job before its time limit with a Unix signal
(`#SBATCH --signal=USR1@60` → `SIGUSR1` 60 s before the limit). The handler only
sets a flag; the main loop sees it at a safe point, saves a checkpoint and exits
cleanly:

```
SLURM → SIGUSR1 → handler sets flag → main loop saves checkpoint → clean exit
```

`--checkpoint-every 0` disables periodic checkpoints, so the checkpoint below
comes only from the signal. The job requests 3 minutes and would need far
longer to finish, so the signal arrives after about 2 minutes.

1. Clean up and submit the first run (`pi_signal.sbatch`):

   ```bash
   rm -f checkpoints/pi_signal.pkl checkpoints/pi_signal.pkl.tmp
   sbatch pi_signal.sbatch
   ```

2. Follow the job (replace `JOBID` with the job ID printed by `sbatch`):

   ```bash
   squeue -u $USER
   tail -f pi_signal_JOBID.out
   ```

3. After about 2 minutes the signal arrives and the job exits cleanly:

   ```
   ...
   darts =  342,000,000 / 10,000,000,000   pi = 3.14166802   error = 7.537e-05   elapsed = 91.39 s

   ========================================================================
   Termination signal received
   ========================================================================
   Signal             : SIGUSR1
   Completed darts    : 342,615,676
   Saving checkpoint ...
   Checkpoint saved   : checkpoints/pi_signal.pkl
   Exiting cleanly.
   ========================================================================
   ```

4. Confirm the checkpoint exists, then submit the restart script
   (`pi_signal_restart.sbatch`). It is identical except for `--resume`:

   ```bash
   ls -l checkpoints/pi_signal.pkl
   sbatch pi_signal_restart.sbatch
   tail -f pi_signal_restart_JOBID.out
   ```

   ```
   ========================================================================
   Restarting from checkpoint
   ========================================================================
   Completed darts    : 342,615,676
   Inside circle      : 269,096,141
   Previous runtime   : 91.57 s
   ========================================================================
   ```

   The restarted job also runs until its own `SIGUSR1`, saving a new
   checkpoint. Repeat step 4 as needed.

### 4. Automatic restart — `pi_requeue.py` (batch)

SLURM restarts a requeued job from the top of the batch script, so
`pi_requeue.py` looks for a checkpoint on startup and loads it if present — no
`--resume` needed. `SLURM_RESTART_COUNT` (0 on the first run, incremented on
each requeue) makes restarts visible in the log. `--requeue` permits requeueing
and `--open-mode=append` keeps the output file across restarts.

`pi_requeue.sbatch` also triggers the requeue itself, so the whole cycle runs
without manual steps. `#SBATCH --signal=B:USR1@60` sends `SIGUSR1` to the batch
shell 60 s before the 3-minute time limit; a `trap` handler in the script
reports the latest checkpoint and calls `scontrol requeue`. `pi_requeue.py`
receives `SIGTERM` from the requeue and saves a checkpoint before stopping. The
application runs in the background under `srun` so the shell can receive the
signal:

```
SIGUSR1 → batch-script trap → scontrol requeue → job back in queue
    → batch script restarts → checkpoint found → computation continues
```

The calculation (1 billion darts) needs about 5 minutes, so the job is requeued
about twice before it completes. The script runs in `rc-testing` so the short
time limit shows this quickly; for real runs use `--partition=serial_requeue`,
where SLURM also requeues the job on preemption.

1. Clean up and submit `pi_requeue.sbatch`:

   ```bash
   rm -f checkpoints/pi_requeue.pkl checkpoints/pi_requeue.pkl.tmp
   JOBID=$(sbatch --parsable pi_requeue.sbatch)
   squeue -j $JOBID
   tail -f pi_requeue_${JOBID}.out
   ```

   ```
   Restart count : 0
   ...
   No checkpoint found.
   Starting a new calculation.
   ```

2. After about 2 minutes the batch script receives `SIGUSR1` and requeues the
   job:

   ```
   SIGUSR1 received: Wed Sep 23 18:43:07 EDT 2026
   Preparing to requeue job 15378
   Latest checkpoint:
   -rw-r--r--. 1 pkrastev rc_admin 3.9K Sep 23 18:43 checkpoints/pi_requeue.pkl
   Requesting requeue...
   ```

3. The job returns to the queue. When it runs again, the same output file shows
   the automatic restart:

   ```
   Slurm job ID        : 15378
   Slurm restart count : 1

   ========================================================================
   CHECKPOINT FOUND - AUTOMATIC RESTART
   ========================================================================
   Completed darts  : 370,686,091
   Inside circle    : 291,139,215
   Previous runtime : 106.07 s
   ========================================================================
   ```

4. The cycle repeats until the calculation completes. Confirm the number of
   restarts:

   ```bash
   sacct -j $JOBID -X -o JobID,State,Elapsed,Restarts
   ```

   ```
   ========================================================================
   CALCULATION COMPLETE
   ========================================================================
   ...
   Application completed normally
   Restart count : 2
   ```

   To requeue by hand instead, run `scontrol requeue $JOBID` while the job is
   running.

> **Do not add `--fresh` to `pi_requeue.sbatch`.** Every requeue reruns the
> script from the top, so a hardcoded `--fresh` would discard the checkpoint the
> restarted job needs. To start a genuinely new run, delete the checkpoint
> *before the initial submission*.

### Cleanup

```bash
rm -f checkpoints/pi_*.pkl checkpoints/pi_*.pkl.tmp
rm -f pi_signal_*.out pi_signal_*.err pi_signal_restart_*.out pi_signal_restart_*.err
rm -f pi_requeue_*.out pi_requeue_*.err
```
