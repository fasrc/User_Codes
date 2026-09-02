# Exercise 3 — Transparent Checkpoint/Restart with DMTCP

## Introduction

This exercise introduces **transparent checkpoint/restart** using [DMTCP](http://dmtcp.sourceforge.net/) - a transparent checkpoint-restart (C/R) tool that can preserve the state of an arbitrary threaded or distributed application to disk
for the purpose of resuming it at a later time or in a different location.
Being "transparent" describes how this process requires no modifications to either the application code or the Linux kernel. DMTCP manages the checkpoint/restart process externally.

In this example we use a C application `long_run.c` with no checkpoint logic to illustrate **transparent** checkpoint with DMTCP.

DMTCP supports a variety of applications, frameworks and programming languages including OpenMP, MATLAB, Python, C, C++, Fortran, shell scripting languages, and workflow management tools.  

## Content

| File | Description |
|---|---|
| `long_run.c` | Simple long-running C program with no checkpoint logic |
| `Makefile` | Builds the example program |
| `dmtcp_checkpoint.sbatch` | Runs the application under DMTCP and creates periodic checkpoints |
| `dmtcp_restart.sbatch` | Restarts the application from the most recent DMTCP checkpoint |
| `dmtcp_auto_requeue.sbatch` | Optional: automated checkpoint/requeue/restart example |

---

## Setup

Load the compiler and DMTCP modules:

```bash
module load gcc/15.2.0-fasrc01
module load dmtcp/4.1.0-fasrc01
```

>**Note:** For compatibility and reproducibility, use a specific `DMTCP` version.

Build the example using the provided `Makefile`:

```bash
make
```

This creates:

```text
long_run.x
```

>**Note:** Using DMTCP to checkpoint and restart applications does not 
require code modifications, but, it *does* require that applications be dynamically linked and use shared libraries (`*.so` files) instead of static libraries.

Create a checkpoint directory:

```bash
mkdir -p checkpoints
```

# 1. Interactive DMTCP Example

Start an interactive allocation:

```bash
salloc --partition=test --time=00:30:00 --cpus-per-task=1 --mem=1G
```

Load the modules:

```bash
module load gcc/15.2.0-fasrc01
module load dmtcp/4.1.0-fasrc01
```

## Run without DMTCP

First run the application normally:

```bash
./long_run.x --iterations 100 --sleep-ms 1000
```

You should see something similar to the below:

```text
$ ./long_run.x --iterations 100 --sleep-ms 1000
============================================================
DMTCP long-running demo application
============================================================
PID            : 3155211
Host           : builds01.rc.fas.harvard.edu
Iterations     : 100
Sleep/iter     : 1000 ms
Checkpoint code: NONE
============================================================
iteration =    1   accumulator = 1.00000000
iteration =    2   accumulator = 2.41421356
iteration =    3   accumulator = 4.14626437
iteration =    4   accumulator = 6.14626437
iteration =    5   accumulator = 8.38233235
...
```

Interrupt it with:

```text
Ctrl-C
```

Run it again:

```bash
./long_run --iterations 100 --sleep-ms 1000
```

It starts again from iteration 1.

## Run under DMTCP

Start the same unchanged application with:

```bash
dmtcp_launch ./long_run.x --iterations 100 --sleep-ms 1000
```

Let it run for a short time.

From another shell in the same allocation, request a checkpoint:

```bash
module load dmtcp/4.1.0-fasrc01
dmtcp_command --checkpoint
```

DMTCP creates checkpoint files such as:

```text
ckpt_long_run_*.dmtcp
dmtcp_restart_script.sh
```

Inspect them:

```bash
ls -lh *.dmtcp
ls -l dmtcp_restart_script.sh
```

Allow the application to continue for a few more iterations, then terminate it:

```bash
dmtcp_command --kill
```

Restart from the checkpoint:

```bash
./dmtcp_restart_script.sh
```

The application resumes from approximately the checkpointed iteration rather than starting again from iteration 1.

### Key point

The source code in `long_run.c` was not modified.

```text
Application-level checkpointing:

application -> save selected state -> restart logic


DMTCP:

application -> process snapshot -> transparent restart
```

# 2. Batch Checkpoint Job

The first batch script starts the application under DMTCP and creates periodic checkpoints.

## `dmtcp_checkpoint.sbatch`

```bash
#!/bin/bash

#SBATCH --job-name=dmtcp-ckpt
#SBATCH --partition=test
#SBATCH --time=00:03:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

#SBATCH --output=dmtcp_ckpt_%j.out
#SBATCH --error=dmtcp_ckpt_%j.err

set -euo pipefail

module load dmtcp

mkdir -p checkpoints
cd checkpoints

PORT_FILE="${SLURM_TMPDIR:-/tmp}/dmtcp_port_${SLURM_JOB_ID}"
rm -f "$PORT_FILE"

dmtcp_coordinator \
    --daemon \
    --port 0 \
    --port-file "$PORT_FILE" \
    --interval 60

while [[ ! -s "$PORT_FILE" ]]; do
    sleep 0.1
done

export DMTCP_COORD_HOST="$(hostname)"
export DMTCP_COORD_PORT="$(cat "$PORT_FILE")"

echo "============================================================"
echo "DMTCP checkpoint job"
echo "============================================================"
echo "Job ID           : ${SLURM_JOB_ID}"
echo "Node             : $(hostname)"
echo "Coordinator port : ${DMTCP_COORD_PORT}"
echo "Start            : $(date)"
echo "============================================================"

dmtcp_launch \
    --join-coordinator \
    ../long_run \
    --iterations 600 \
    --sleep-ms 1000
```

The application runs for about 10 minutes, but the Slurm job requests only 3 minutes.

DMTCP creates a checkpoint every 60 seconds:

```text
start
  |
  +---- 60 s ---- CKPT
  |
  +---- 60 s ---- CKPT
  |
  +---- wall time ---- X
```

Submit:

```bash
sbatch dmtcp_checkpoint.sbatch
```

Monitor:

```bash
squeue -u $USER
```

After the job ends, inspect the checkpoint directory:

```bash
ls -lh checkpoints/
```

You should see files similar to:

```text
ckpt_long_run_*.dmtcp
dmtcp_restart_script.sh
```

---

# 3. Batch Restart Job

The restart job starts a new Slurm allocation and resumes from the most recent successful DMTCP checkpoint.

## `dmtcp_restart.sbatch`

```bash
#!/bin/bash

#SBATCH --job-name=dmtcp-restart
#SBATCH --partition=test
#SBATCH --time=00:03:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

#SBATCH --output=dmtcp_restart_%j.out
#SBATCH --error=dmtcp_restart_%j.err

set -euo pipefail

module load dmtcp

cd checkpoints

if [[ ! -f dmtcp_restart_script.sh ]]; then
    echo "ERROR: dmtcp_restart_script.sh not found."
    echo "Run dmtcp_checkpoint.sbatch first."
    exit 1
fi

PORT_FILE="${SLURM_TMPDIR:-/tmp}/dmtcp_port_${SLURM_JOB_ID}"
rm -f "$PORT_FILE"

dmtcp_coordinator \
    --daemon \
    --port 0 \
    --port-file "$PORT_FILE" \
    --interval 60

while [[ ! -s "$PORT_FILE" ]]; do
    sleep 0.1
done

export DMTCP_COORD_HOST="$(hostname)"
export DMTCP_COORD_PORT="$(cat "$PORT_FILE")"

echo "============================================================"
echo "DMTCP restart job"
echo "============================================================"
echo "Job ID           : ${SLURM_JOB_ID}"
echo "Node             : $(hostname)"
echo "Coordinator port : ${DMTCP_COORD_PORT}"
echo "Restart          : $(date)"
echo "============================================================"

./dmtcp_restart_script.sh
```

Submit:

```bash
sbatch dmtcp_restart.sbatch
```

The application resumes from the most recent checkpoint created by the previous job.

For example:

```text
Checkpoint job:

iteration = 1
...
iteration = 60
...
iteration = 120
...
job ends


Restart job:

iteration ≈ 121
iteration ≈ 122
...
```

The restart job is a new Slurm job and may run on a different compute node.

---

# 4. Optional: Automated Checkpoint/Restart with Requeue

The previous two examples separate checkpointing and restart into two batch jobs.

An automated workflow can combine:

```text
launch
  |
  v
periodic checkpoints
  |
  v
Slurm warning signal
  |
  v
final checkpoint
  |
  v
requeue
  |
  v
restart from latest checkpoint
```

This is useful for longer-running workflows, but it is optional for this exercise.

The example script is:

```text
dmtcp_auto_requeue.sbatch
```

It is designed for the CANNON `serial_requeue` partition and uses:

```bash
#SBATCH --partition=serial_requeue
#SBATCH --signal=B:USR1@60
#SBATCH --requeue
#SBATCH --open-mode=append
```

The first execution launches the application with `dmtcp_launch`.

After requeue, the same batch script starts again and uses:

```bash
./dmtcp_restart_script.sh
```

to continue from the latest successful checkpoint.

The batch shell catches `SIGUSR1`, requests a final checkpoint, and requeues the job.

This optional example demonstrates the transition from:

```text
manual checkpoint/restart
```

to:

```text
automated checkpoint/restart
```

---

## Comparing Application-Level Checkpointing and DMTCP

| Application checkpointing | DMTCP |
|---|---|
| Application explicitly saves state | Application does not need checkpoint code |
| Saves selected variables | Saves process/runtime state |
| Usually smaller checkpoints | Usually larger checkpoints |
| Requires source-code changes | No source-code changes |
| Application controls restart logic | DMTCP reconstructs process state |

In Exercises 1 and 2, we asked:

> What application state is required to continue the calculation?

With DMTCP, the question becomes:

> What process state must be restored to recreate the running application?

---

## Key Concepts

- **Transparent checkpoint/restart** — the application does not need checkpoint-specific source code.

- **DMTCP coordinator** — coordinates checkpointing and restart for DMTCP-managed processes.

- **`dmtcp_launch`** — starts an application under DMTCP control.

- **`dmtcp_command --checkpoint`** — requests a checkpoint.

- **`dmtcp_restart_script.sh`** — restarts from the latest successful checkpoint set.

- **Periodic checkpointing** — the coordinator can create checkpoints automatically at a specified interval.

- **Checkpoint interval** — work completed after the last checkpoint but before termination may need to be repeated.

- **Batch restart** — a new Slurm job can restart a previously checkpointed application.

---

## Cleanup

Remove generated checkpoint files:

```bash
rm -rf checkpoints/*
```

Remove Slurm logs:

```bash
rm -f dmtcp_ckpt_*.out
rm -f dmtcp_ckpt_*.err

rm -f dmtcp_restart_*.out
rm -f dmtcp_restart_*.err

rm -f dmtcp_auto_*.out
rm -f dmtcp_auto_*.err
```

Rebuild:

```bash
make clean
make
```

---

## Summary

The progression in this exercise is:

```text
Normal application
       |
       v
DMTCP interactive checkpoint/restart
       |
       v
Batch checkpoint job
       |
       v
Batch restart job
       |
       v
Optional automated requeue workflow
```

The main lesson is:

> **DMTCP provides transparent checkpoint/restart without requiring application-level checkpoint code.**

