#!/bin/bash
#SBATCH --job-name=mana-pi-restart
#SBATCH --partition=rc-testing
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=00:02:00
#SBATCH --mem-per-cpu=1G
#SBATCH --output=mana_restart_%j.out
#SBATCH --error=mana_restart_%j.err

set -euo pipefail

source /n/holylabs/rc_admin/Everyone/checkpoint-training/setup_mana.sh

cd "${SLURM_SUBMIT_DIR}/checkpoints"

if [[ ! -d ckpt_rank_0 ]]; then
    echo "ERROR: No MANA checkpoint found in:"
    echo "       ${SLURM_SUBMIT_DIR}/checkpoints"
    echo "Run mana_checkpoint.sbatch first."
    exit 1
fi

# Continue creating checkpoints every 30 seconds after restart.
mana_coordinator -i30

echo "============================================================"
echo "MANA restart job"
echo "============================================================"
echo "Job ID  : ${SLURM_JOB_ID}"
echo "Node    : $(hostname)"
echo "Restart : $(date)"
echo "============================================================"

# Run from inside the checkpoint directory. This matches the
# interactive workflow that successfully avoided the current
# mana_restart relative-path issue.
srun --mpi=pmix -n "${SLURM_NTASKS}" \
    mana_restart
