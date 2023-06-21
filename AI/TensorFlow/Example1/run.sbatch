#!/bin/bash
#SBATCH -p gpu_test
#SBATCH -n 1
#SBATCH -t 0-03:00
#SBATCH -J dnn
#SBATCH -o tf_mnist.out
#SBATCH -e tf_mnist.err
#SBATCH --gres=gpu:1
#SBATCH --mem=8G

# --- Set up software environment ---
module load python/3.10.9-fasrc01
source activate tf2.12_cuda11

# --- Run the code ---
srun -n 1 --gres=gpu:1 python tf_mnist.py
