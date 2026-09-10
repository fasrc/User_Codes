#!/bin/bash
#SBATCH --job-name=mana-pi-auto
#SBATCH --partition=rc-testing
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=00:02:00
#SBATCH --mem-per-cpu=1G
#SBATCH --output=mana_auto_%j.out
#SBATCH --error=mana_auto_%j.err
#SBATCH --signal=B:USR1@30
#SBATCH --requeue
#SBATCH --open-mode=append

set -uo pipefail

source /n/holylabs/rc_admin/Everyone/checkpoint-training/setup_mana.sh

SUBMIT_DIR="${SLURM_SUBMIT_DIR}"
CHECKPOINT_DIR="${SUBMIT_DIR}/checkpoints_auto"
RESTART_COUNT="${SLURM_RESTART_COUNT:-0}"

echo "============================================================"
echo "MANA automatic checkpoint/restart"
echo "============================================================"
echo "Job ID        : ${SLURM_JOB_ID}"
echo "Restart count : ${RESTART_COUNT}"
echo "Node          : $(hostname)"
echo "Start         : $(date)"
echo "============================================================"

checkpoint_and_requeue()
{
    trap - USR1

    echo
    echo "USR1 received at $(date)"
    echo "Requesting a final MANA checkpoint..."

    if mana_status --checkpoint; then
        echo "Checkpoint completed at $(date)"
    else
        echo "ERROR: MANA checkpoint request failed."
        exit 1
    fi

    echo "Requeueing Slurm job ${SLURM_JOB_ID}..."
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
}

trap checkpoint_and_requeue USR1

if [[ "${RESTART_COUNT}" -eq 0 ]]; then
    echo "First execution: starting a new MPI calculation."

    cd "${SUBMIT_DIR}"
    rm -rf "${CHECKPOINT_DIR}"
    mkdir -p "${CHECKPOINT_DIR}"

    # Periodic checkpoints every 60 seconds.
    mana_coordinator -i60

    srun --mpi=pmix -n "${SLURM_NTASKS}" \
        mana_launch --ckptdir "${CHECKPOINT_DIR}" \
        ./pi_mpi_mana.x \
        --darts 3000000000 \
        --report-every 5000000 \
        --sleep-ms 1000 &

else
    echo "Requeued execution: restarting from the latest MANA checkpoint."

    if [[ ! -d "${CHECKPOINT_DIR}/ckpt_rank_0" ]]; then
        echo "ERROR: No MANA checkpoint found in:"
        echo "       ${CHECKPOINT_DIR}"
        exit 1
    fi

    cd "${CHECKPOINT_DIR}"

    # Continue periodic checkpointing after restart.
    mana_coordinator -i60

    srun --mpi=pmix -n "${SLURM_NTASKS}" \
        mana_restart &
fi

APP_PID=$!

# Keep the batch shell alive so SIGUSR1 is delivered here.
wait "${APP_PID}"
status=$?

echo
echo "Application finished with status ${status} at $(date)"
exit "${status}"
