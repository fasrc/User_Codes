#!/bin/bash
#SBATCH -p gpu_test
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --mem=12000
#SBATCH -J saxpy_nb_test
#SBATCH -o saxpy_nb_test.out
#SBATCH -e saxpy_nb_test.err
#SBATCH -t 30

# Load required modules
module load python/3.10.12-fasrc01
module load cuda/12.2.0-fasrc01
source activate numba_env

# Run the program
srun -n 1 -c 1 python saxpy_np.py
