#!/bin/bash
#SBATCH --job-name=pi-requeue
#SBATCH --partition=rc-testing
#SBATCH --time=00:03:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=pi_requeue_%j.out
#SBATCH --error=pi_requeue_%j.err
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:USR1@60

set -euo pipefail

module load python

cd "${SLURM_SUBMIT_DIR}"

mkdir -p checkpoints

CHECKPOINT_FILE="checkpoints/pi_requeue.pkl"

echo "============================================================"
echo "Starting Python automatic-requeue example"
echo "============================================================"
echo "Job ID        : ${SLURM_JOB_ID}"
echo "Restart count : ${SLURM_RESTART_COUNT:-0}"
echo "Node          : $(hostname)"
echo "Time          : $(date)"
echo "============================================================"


# ---------------------------------------------------------------------------
# Requeue handler
#
# Slurm sends SIGUSR1 to the batch shell before the time limit.
# pi_requeue.py already creates periodic checkpoints, so the handler
# does not need to create a new checkpoint itself. It simply requests
# that Slurm requeue the job.
# ---------------------------------------------------------------------------

requeue_job()
{
    # Prevent the handler from being called a second time.
    trap - USR1

    echo
    echo "============================================================"
    echo "SIGUSR1 received: $(date)"
    echo "Preparing to requeue job ${SLURM_JOB_ID}"
    echo "============================================================"

    if [[ -f "${CHECKPOINT_FILE}" ]]; then
        echo "Latest checkpoint:"
        ls -lh "${CHECKPOINT_FILE}"
    else
        echo "WARNING: no checkpoint exists yet."
    fi

    echo "Requesting requeue..."

    scontrol requeue "${SLURM_JOB_ID}"

    exit 0
}

trap requeue_job USR1


# ---------------------------------------------------------------------------
# Run the application.
#
# No --resume is needed: pi_requeue.py automatically loads the checkpoint
# if one exists.
#
# Do NOT add --fresh here. On a requeued run, --fresh would discard the
# checkpoint and restart the computation from the beginning.
#
# Run srun in the background so that the batch shell can receive SIGUSR1
# and execute the trap above.
# ---------------------------------------------------------------------------

srun python pi_requeue.py \
    --darts 1000000000 \
    --seed 42 \
    --report-every 1000000 \
    --checkpoint-every 5000000 \
    --checkpoint-file "${CHECKPOINT_FILE}" &

APP_PID=$!

wait "${APP_PID}"


# ---------------------------------------------------------------------------
# If we reach this point, the application completed normally before the
# Slurm warning signal was received.
# ---------------------------------------------------------------------------

echo
echo "============================================================"
echo "Application completed normally"
echo "============================================================"
echo "Job ID        : ${SLURM_JOB_ID}"
echo "Restart count : ${SLURM_RESTART_COUNT:-0}"
echo "Time          : $(date)"
echo "============================================================"
