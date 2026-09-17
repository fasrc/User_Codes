#!/bin/bash
#SBATCH --job-name=gmx-checkpoint
#SBATCH --partition=rc-testing
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=00:02:00
#SBATCH --mem-per-cpu=1G
#SBATCH --output=gmx_checkpoint_%j.out
#SBATCH --error=gmx_checkpoint_%j.err

set -euo pipefail

#source /n/holylabs/rc_admin/Everyone/checkpoint-training/gromacs/sw/bin/GMXRC
source gromacs_setup.sh 
cd "${SLURM_SUBMIT_DIR}"

[[ -f md.tpr ]] || { echo "ERROR: md.tpr not found."; exit 1; }

if [[ -f md.cpt ]]; then
    echo "ERROR: md.cpt already exists."
    echo "Use gromacs_restart.sbatch, or clean/rename the old md.* files."
    exit 1
fi

echo "Job ID : ${SLURM_JOB_ID}"
echo "Node   : $(hostname)"
echo "Start  : $(date)"

srun --mpi=pmix -n "${SLURM_NTASKS}" \
    gmx_mpi mdrun \
    -deffnm md \
    -cpt 0.1 \
    -maxh 0.025

echo "Stopped/finished: $(date)"
ls -lh md.cpt

