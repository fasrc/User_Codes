#!/bin/bash
#SBATCH --job-name=mana-restart
#SBATCH --partition=rc-testing
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=00:02:00
#SBATCH --mem-per-cpu=1G
#SBATCH --output=mana_restart_%j.out
#SBATCH --error=mana_restart_%j.err

set -euo pipefail

source /n/holylabs/rc_admin/Everyone/checkpoint-training/setup_mana.sh

CHECKPOINT_DIR="${SLURM_SUBMIT_DIR}/checkpoints"

echo "============================================================"
echo "MANA restart"
echo "============================================================"
echo "Job ID          : ${SLURM_JOB_ID}"
echo "Submit directory: ${SLURM_SUBMIT_DIR}"
echo "Checkpoint dir  : ${CHECKPOINT_DIR}"
echo "MPI tasks       : ${SLURM_NTASKS}"
echo "Node            : $(hostname)"
echo "Time            : $(date)"
echo "============================================================"

# ---------------------------------------------------------------------------
# Verify checkpoint directory
# ---------------------------------------------------------------------------

if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
    echo "ERROR: checkpoint directory does not exist:"
    echo "       ${CHECKPOINT_DIR}"
    exit 1
fi

echo
echo "Checkpoint directories:"
ls -ld "${CHECKPOINT_DIR}"/ckpt_rank_* || {
    echo "ERROR: no ckpt_rank_* directories found."
    exit 1
}

NUM_RANK_DIRS=$(
    find "${CHECKPOINT_DIR}" \
        -maxdepth 1 \
        -type d \
        -name 'ckpt_rank_*' |
    wc -l
)

echo
echo "Found ${NUM_RANK_DIRS} checkpoint rank directories."

if [[ "${NUM_RANK_DIRS}" -ne "${SLURM_NTASKS}" ]]; then
    echo "ERROR:"
    echo "  Checkpoint ranks : ${NUM_RANK_DIRS}"
    echo "  Requested ranks  : ${SLURM_NTASKS}"
    echo
    echo "Restart with the same number of MPI ranks used"
    echo "when the checkpoint was created."
    exit 1
fi

echo
echo "Checkpoint files:"
find "${CHECKPOINT_DIR}" \
    -maxdepth 2 \
    -type f \
    -ls

# ---------------------------------------------------------------------------
# IMPORTANT:
#
# For the MANA installation used in this exercise, restart from inside
# the checkpoint directory.
# ---------------------------------------------------------------------------

cd "${CHECKPOINT_DIR}"

echo
echo "Restart working directory:"
pwd

echo
echo "Starting MANA coordinator..."

mana_coordinator -i15

echo
echo "Restarting MPI application..."

srun --mpi=pmix \
    -n "${SLURM_NTASKS}" \
    mana_restart

echo
echo "============================================================"
echo "MANA restart completed"
echo "Time: $(date)"
echo "============================================================"
