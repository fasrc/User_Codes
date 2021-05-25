#!/bin/bash
#SBATCH -p gpu_test
#SBATCH -n 1
#SBATCH -t 0-03:00
#SBATCH -J dnn
#SBATCH -o tf_mnist.out
#SBATCH -e tf_mnist.err
#SBATCH --gres=gpu:1
#SBATCH --gpu-freq=high
#SBATCH --mem=8G

# --- Set up software environment ---
module load python/3.8.5-fasrc01 
module load cuda/11.1.0-fasrc01
module load cudnn/8.1.0.77_cuda11.2-fasrc01
source activate tf2.5_cuda11

# --- Run the code ---
srun -n 1 --gres=gpu:1 python tf_mnist.py
