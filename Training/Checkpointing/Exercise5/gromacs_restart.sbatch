#!/bin/bash
#SBATCH --job-name=gmx-restart
#SBATCH --partition=rc-testing
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=00:02:00
#SBATCH --mem-per-cpu=1G
#SBATCH --output=gmx_restart_%j.out
#SBATCH --error=gmx_restart_%j.err

set -euo pipefail

#source /n/holylabs/rc_admin/Everyone/checkpoint-training/gromacs/sw/bin/GMXRC
source gromacs_setup.sh
cd "${SLURM_SUBMIT_DIR}"

[[ -f md.tpr ]] || { echo "ERROR: md.tpr not found."; exit 1; }
[[ -f md.cpt ]] || { echo "ERROR: md.cpt not found. Run gromacs_checkpoint.sbatch first."; exit 1; }

echo "Job ID  : ${SLURM_JOB_ID}"
echo "Node    : $(hostname)"
echo "Restart : $(date)"

srun --mpi=pmix -n "${SLURM_NTASKS}" \
    gmx_mpi mdrun \
    -deffnm md \
    -cpi md.cpt \
    -cpt 0.1 \
    -maxh 0.025

echo "Stopped/finished: $(date)"
ls -lh md.cpt

