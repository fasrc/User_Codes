# Exercise 2 — Checkpointing in C

## Introduction

This exercise repeats Exercise 1 in C, where details that Python hides become
explicit: the layout of the saved state, the file writes, and the signal
handling. The pattern is the same:

```
compute → checkpoint → compute → [interruption] → restart → load checkpoint → continue
```

The vehicle is again a Monte Carlo estimate of $\pi$: random points $(x, y)$ are
thrown into a unit square, and the fraction landing inside the unit circle
($x^2 + y^2 \le 1$) converges to $\pi / 4$:

$$
\pi \approx 4 \times \frac{\text{points inside circle}}{\text{total points}}
$$

The same calculation is hardened in four steps: no checkpointing, periodic
checkpointing, signal-aware checkpointing, and automatic restart on the CANNON
`serial_requeue` partition. The first two run **interactively**, the last two
in **batch** mode.

Checkpointing is done at the application level: the program periodically writes
a C structure (`checkpoint_t`) to a binary file with `fwrite()`, and on restart
reads it back and continues from the next dart. The checkpoint holds a magic
string and version number, the number of completed darts, the number inside the
circle, the random-number generator state, the total darts and seed (validated
on resume), and the elapsed runtime.

Key ideas along the way:

- **Explicit RNG state** — the `xorshift64*` generator used here has a single
  `uint64_t` state, so a resumed run reproduces the exact random sequence of an
  uninterrupted run.
- **Atomic writes** — write to a temporary file, `fflush()`, `fsync()`,
  `fclose()`, then `rename()`, so a crash mid-write never corrupts the last good
  checkpoint.
- **Signal-safe handlers** — the handler only sets `volatile sig_atomic_t`
  flags; all checkpoint I/O happens later in the main loop.
- **Portability** — writing a raw structure is simple but not a portable
  production format (padding, byte order, layout changes); real applications
  often use an explicit format or a library such as HDF5.

---

## Content

| File | Description | Mode |
|------|-------------|------|
| `pi_naive.c` | Baseline, no checkpointing | Interactive |
| `pi_checkpoint.c` | Periodic, atomic binary checkpoints; manual `--resume` | Interactive |
| `pi_signal.c` | Adds `SIGUSR1`/`SIGTERM` handling that checkpoints on scheduler warning | Batch |
| `pi_signal.sbatch` | Starts a new `pi_signal` run with `--signal=USR1@60` | Batch |
| `pi_signal_restart.sbatch` | Restarts `pi_signal` from its checkpoint (`--resume`) | Batch |
| `pi_c_signal_15366.out` | Example output of the first `pi_signal.sbatch` run (stopped by `SIGUSR1` at 106,000,001 darts) | Batch |
| `pi_c_signal_restart_15367.out` | Example output of the `pi_signal_restart.sbatch` run (resumed from that checkpoint, stopped by `SIGUSR1` at 223,000,001 darts) | Batch |
| `pi_requeue.c` | Finds an existing checkpoint automatically on startup | Batch |
| `pi_requeue.sbatch` | Runs `pi_requeue` with `--requeue`; traps `SIGUSR1` (`--signal=B:USR1@60`) and requeues itself | Batch |
| `pi_c_requeue_15380.out` | Example output of a `pi_requeue.sbatch` run that was requeued twice (restarted from checkpoints at 98,000,001 and 188,000,001 darts) and then completed | Batch |
| `Makefile` | Builds the four programs (`make clean` removes them and the checkpoints) | — |

---

## Workflow

### Setup

Steps 1 and 2 run on a compute node, not a login node. The programs must be
built before any batch job is submitted:

```bash
salloc --partition=test --time=00:20:00 --cpus-per-task=1 --mem=1G
module load gcc
make
mkdir -p checkpoints
```

`make` creates `pi_naive`, `pi_checkpoint`, `pi_signal` and `pi_requeue`. In all
examples `--sleep 1` slows the loop so there is time to interrupt it.

### 1. No checkpointing — `pi_naive.c` (interactive)

Progress lives only in process memory, so an interruption loses everything.

1. Start the calculation:

   ```bash
   ./pi_naive \
       --darts 50000000 \
       --seed 42 \
       --report-every 1000000 \
       --sleep 1
   ```

   ```
   ...
   darts =      1000000 / 50000000   pi = 3.14330800   error = 1.715e-03   elapsed = 0.01 s
   darts =      2000000 / 50000000   pi = 3.14220000   error = 6.073e-04   elapsed = 1.02 s
   darts =      3000000 / 50000000   pi = 3.14184267   error = 2.500e-04   elapsed = 2.04 s
   darts =      4000000 / 50000000   pi = 3.14236800   error = 7.753e-04   elapsed = 3.05 s
   darts =      5000000 / 50000000   pi = 3.14191920   error = 3.265e-04   elapsed = 4.07 s
   ...
   ```

2. Press `Ctrl-C` after a few progress reports.
3. Rerun the same command. It starts again from dart 1 — nothing was saved.

### 2. Periodic checkpointing — `pi_checkpoint.c` (interactive)

Every `--checkpoint-every` darts the program saves the `checkpoint_t` structure
to a binary file. On restart, `--resume` loads it and continues with the next
dart.

1. Remove any old checkpoint:

   ```bash
   rm -f checkpoints/pi_checkpoint.bin checkpoints/pi_checkpoint.bin.tmp
   ```

2. Start the calculation:

   ```bash
   ./pi_checkpoint \
       --darts 50000000 \
       --seed 42 \
       --report-every 1000000 \
       --checkpoint-every 5000000 \
       --checkpoint-file checkpoints/pi_checkpoint.bin \
       --sleep 1
   ```

   ```
   ...
   darts =      5000000 / 50000000   pi = 3.14191920   error = 3.265e-04   elapsed = 4.07 s
     --> checkpoint saved at dart 5000000
   darts =      6000000 / 50000000   pi = 3.14158200   error = 1.065e-05   elapsed = 5.09 s
   ...
   ```

3. After at least two checkpoints (10,000,000 darts), inspect the file
   (`ls -l checkpoints/pi_checkpoint.bin`) and press `Ctrl-C`. Work done after
   the last checkpoint is lost, for example darts 10,000,001–13,000,000 if the
   interruption comes at 13 million.
4. Resume with the same parameters plus `--resume`:

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

   ```
   ========================================================================
   Restarting from checkpoint
   ========================================================================
   Completed darts  : 10000000
   Inside circle    : 7854138
   Previous runtime : 10.15 s
   ========================================================================

   darts =     11000000 / 50000000   pi = 3.14182327   error = 2.306e-04   elapsed = 10.16 s
   ...
   ```

   The run continues from the checkpoint instead of dart 1, and the values match
   the uninterrupted run (the `pi` at 11,000,000 darts is identical). The
   checkpoint must match `--darts` and `--seed`, otherwise the program refuses to
   resume.

### 3. Signal-aware checkpointing — `pi_signal.c` (batch)

SLURM can warn a job before its time limit with a Unix signal
(`#SBATCH --signal=USR1@60` → `SIGUSR1` 60 s before the limit). The program
installs handlers for `SIGUSR1` and `SIGTERM` with `sigaction()`. A handler only
sets a flag; the main loop sees it at a safe point between darts, saves a
checkpoint and exits cleanly:

```
SLURM → SIGUSR1 → handler sets flag → main loop saves checkpoint → clean exit
```

`--checkpoint-every 0` disables periodic checkpoints, so the checkpoint below
comes only from the signal. Without throttling, the C program would finish
10 billion darts in under a minute, so `--sleep 1` paces it at about
one million darts per second. The job then cannot finish within its 3 minutes,
and the signal arrives after about 2 minutes.

1. Clean up and submit the first run (`pi_signal.sbatch`):

   ```bash
   rm -f checkpoints/pi_signal.bin checkpoints/pi_signal.bin.tmp
   sbatch pi_signal.sbatch
   ```

2. Follow the job (replace `JOBID` with the job ID printed by `sbatch`):

   ```bash
   squeue -u $USER
   tail -f pi_c_signal_JOBID.out
   ```

3. After about 2 minutes the signal arrives and the job exits cleanly:

   ```
   ...
   darts =    ... / 10000000000   pi = ...   error = ...   elapsed = ... s

   Signal received: SIGUSR1
   Checkpointing at dart ...
   Checkpoint saved: checkpoints/pi_signal.bin
   Exiting cleanly.
   ```

4. Confirm the checkpoint exists, then submit the restart script
   (`pi_signal_restart.sbatch`). It is identical except for `--resume`:

   ```bash
   ls -l checkpoints/pi_signal.bin
   sbatch pi_signal_restart.sbatch
   tail -f pi_c_signal_restart_JOBID.out
   ```

   ```
   ========================================================================
   Restarting from checkpoint
   ========================================================================
   Completed darts  : ...
   Inside circle    : ...
   Previous runtime : ... s
   ========================================================================
   ```

   The restarted job also runs until its own `SIGUSR1`, saving a new
   checkpoint. Repeat step 4 as needed.

### 4. Automatic restart — `pi_requeue.c` (batch)

SLURM restarts a requeued job from the top of the batch script, so `pi_requeue`
looks for a checkpoint on startup and loads it if present — no `--resume`
needed. `SLURM_RESTART_COUNT` (0 on the first run, incremented on each requeue)
makes restarts visible in the log. `--requeue` permits requeueing and
`--open-mode=append` keeps the output file across restarts.

`pi_requeue.sbatch` also triggers the requeue itself, so the whole cycle runs
without manual steps. `#SBATCH --signal=B:USR1@60` sends `SIGUSR1` to the batch
shell 60 s before the 3-minute time limit; a `trap` handler in the script
reports the latest checkpoint and calls `scontrol requeue`. `pi_requeue`
receives `SIGTERM` from the requeue and saves a checkpoint before stopping. The
application runs in the background under `srun` so the shell can receive the
signal:

```
SIGUSR1 → batch-script trap → scontrol requeue → job back in queue
    → batch script restarts → checkpoint found → computation continues
```

The job throws 250 million darts, checkpoints every 5 million, and uses
`--sleep 1` to run at about one million darts per second. It needs about 4
minutes, so it is requeued about twice before it completes. The script runs in
`rc-testing` so the short time limit shows this quickly; for real runs use
`--partition=serial_requeue`, where SLURM also requeues the job on preemption.

1. Clean up and submit `pi_requeue.sbatch`:

   ```bash
   rm -f checkpoints/pi_requeue.bin checkpoints/pi_requeue.bin.tmp
   JOBID=$(sbatch --parsable pi_requeue.sbatch)
   squeue -j $JOBID
   tail -f pi_c_requeue_${JOBID}.out
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
   SIGUSR1 received: Wed Sep 23 21:10:07 EDT 2026
   Preparing to requeue job 15380
   Latest checkpoint:
   -rw-r--r--. 1 pkrastev rc_admin 64 Sep 23 21:10 checkpoints/pi_requeue.bin
   Requesting requeue...
   ```

3. The job returns to the queue. When it runs again, the same output file shows
   the automatic restart:

   ```
   Slurm job ID        : 15380
   Slurm restart count : 1
   Checkpoint file     : checkpoints/pi_requeue.bin
   ========================================================================

   CHECKPOINT FOUND - AUTOMATIC RESTART
   Completed darts  : 98000001
   Inside circle    : 76970270
   Previous runtime : 98.00 s

   darts =     99000000 / 250000000   pi = 3.14165087   error = 5.822e-05   elapsed = 98.01 s
   ```

   The job resumed at dart 98,000,001, not at a periodic checkpoint, because the
   `SIGTERM` sent by the requeue made `pi_requeue` save its current state. After
   an unannounced preemption it would resume from the last periodic checkpoint.

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
make clean
rm -f pi_c_signal_*.out pi_c_signal_*.err pi_c_signal_restart_*.out pi_c_signal_restart_*.err
rm -f pi_c_requeue_*.out pi_c_requeue_*.err
```
