#!/bin/bash
#SBATCH -J mmult
#SBATCH -o mmult.out
#SBATCH -e mmult.err
#SBATCH -p shared
#SBATCH -n 4
#SBATCH -t 0-00:30
#SBATCH --mem-per-cpu=4000

# Set up environment
WORK_DIR=/scratch/${USER}/${SLURM_JOB_ID}
PRO=mmult
mkdir -pv ${WORK_DIR}
cd $WORK_DIR
cp ${SLURM_SUBMIT_DIR}/${PRO}.x .

# Load required modules
module load gcc/8.2.0-fasrc01
module load openmpi/3.1.1-fasrc01

# Run program
srun -n $SLURM_NTASKS --mpi=pmix ./${PRO}.x > ${PRO}.dat

# Copy back the result and clean up
cp *.dat ${SLURM_SUBMIT_DIR}
rm -rf ${WORK_DIR}
