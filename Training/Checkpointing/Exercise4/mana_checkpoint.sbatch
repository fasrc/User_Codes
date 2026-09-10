#!/bin/bash
#SBATCH --job-name=mana-pi-launch
#SBATCH --partition=rc-testing
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=00:02:00
#SBATCH --mem-per-cpu=1G
#SBATCH --output=mana_launch_%j.out
#SBATCH --error=mana_launch_%j.err

set -euo pipefail

source /n/holylabs/rc_admin/Everyone/checkpoint-training/setup_mana.sh

cd "${SLURM_SUBMIT_DIR}"

rm -rf checkpoints
mkdir -p checkpoints

# Create a checkpoint every 30 seconds.
mana_coordinator -i30

echo "============================================================"
echo "MANA checkpoint job"
echo "============================================================"
echo "Job ID : ${SLURM_JOB_ID}"
echo "Node   : $(hostname)"
echo "Start  : $(date)"
echo "============================================================"

# pi_mpi_mana.x must be compiled with mpicc_mana.
# The application is deliberately longer than this allocation.
srun --mpi=pmix -n "${SLURM_NTASKS}" \
    mana_launch --ckptdir checkpoints \
    ./pi_mpi_mana.x \
    --darts 10000000000 \
    --report-every 5000000 \
    --sleep-ms 1000

