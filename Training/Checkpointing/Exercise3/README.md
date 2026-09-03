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
| `dmtcp_auto_15135.out` | Example output from an automated checkpoint/requeue/restart run |


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
./long_run.x --iterations 100 --sleep-ms 1000
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

>**Note:** You can use <br> `scontrol show job <JOBID> | grep -oP 'BatchHost=\K\w+'` <br> to display the node where the job ran and then `ssh` to it.

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
#SBATCH --partition=rc-testing
#SBATCH --time=00:03:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=dmtcp_ckpt_%j.out
#SBATCH --error=dmtcp_ckpt_%j.err

set -euo pipefail

module load dmtcp

mkdir -p checkpoints
cd checkpoints

#
# Use a random coordinator port to avoid collisions with other users.
#
PORT_FILE="${SLURM_TMPDIR:-/tmp}/dmtcp_port_${SLURM_JOB_ID}"
rm -f "$PORT_FILE"

#
# Start a DMTCP coordinator and checkpoint once every 60 seconds.
#
dmtcp_coordinator \
    --daemon \
    --port 0 \
    --port-file "$PORT_FILE" \
    --interval 60

#
# Wait for the coordinator to write its port number.
#
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

#
# The application itself contains NO checkpointing code.
#
dmtcp_launch \
    --join-coordinator \
    ../long_run.x \
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
#SBATCH --partition=rc-testing
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

#
# Start a fresh coordinator for this new allocation.
# Continue making checkpoints every 60 seconds.
#
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

#
# Resume from the most recent successful DMTCP checkpoint.
#
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
## Example output from Automated C/R with Re-queue

```text
$ cat dmtcp_auto_15135.out 
============================================================
DMTCP automatic checkpoint/restart
============================================================
Job ID        : 15135
Restart count : 0
Node          : holy7c26506.rc.fas.harvard.edu
Time          : Thu Sep  3 14:20:44 EDT 2026
============================================================
Starting new DMTCP computation.
============================================================
DMTCP long-running demo application
============================================================
PID            : 40000
Host           : holy7c26506.rc.fas.harvard.edu
Iterations     : 600
Sleep/iter     : 1000 ms
Checkpoint code: NONE
============================================================
iteration =    1   accumulator = 1.00000000
iteration =    2   accumulator = 2.41421356
...
iteration =  113   accumulator = 805.91542424
iteration =  114   accumulator = 816.59250249

USR1 received at Thu Sep  3 14:22:39 EDT 2026
Creating final checkpoint...
iteration =  115   accumulator = 827.31630779
Checkpoint complete.
Requeueing job 15135...
============================================================
DMTCP automatic checkpoint/restart
============================================================
Job ID        : 15135
Restart count : 1
Node          : holy7c26506.rc.fas.harvard.edu
Time          : Thu Sep  3 14:25:07 EDT 2026
============================================================
Restarting DMTCP computation.
iteration =   63   accumulator = 337.13065543
iteration =   64   accumulator = 345.13065543
iteration =   65   accumulator = 353.19291317
...
iteration =  212   accumulator = 2064.91948071
iteration =  213   accumulator = 2079.51400023

USR1 received at Thu Sep  3 14:30:40 EDT 2026
Creating final checkpoint...
Checkpoint complete.
Requeueing job 15135...
============================================================
DMTCP automatic checkpoint/restart
============================================================
Job ID        : 15135
Restart count : 3
Node          : holy7c26506.rc.fas.harvard.edu
Time          : Thu Sep  3 14:33:08 EDT 2026
============================================================
Restarting DMTCP computation.
iteration =  185   accumulator = 1684.11060850
iteration =  186   accumulator = 1697.74879020
iteration =  187   accumulator = 1711.42358453
...
iteration =  596   accumulator = 9712.14323876
iteration =  597   accumulator = 9736.57682221
iteration =  598   accumulator = 9761.03086073
iteration =  599   accumulator = 9785.50533723
iteration =  600   accumulator = 9810.00023466
============================================================
Calculation complete
Final iteration   : 600
Final accumulator : 9810.00023466
============================================================

Application finished with status 0.
End: Thu Sep  3 14:57:59 EDT 2026
```

## References:

* [DMTCP website](http://dmtcp.sourceforge.net/index.html)
* [DMTCP github](https://github.com/dmtcp/dmtcp/blob/master/QUICK-START.md)
* [NERSC: DMTCP user training slides  (Nov. 2019)](https://wayback.archive-it.org/23486/https://www.nersc.gov/user-training-on-checkpointing-and-restarting-jobs-using-dmtcp-on-november-6-2019/)  