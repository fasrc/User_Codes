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
module load python/3.7.7-fasrc01
module load cuda/10.1.243-fasrc01
module load cudnn/7.6.5.32_cuda10.1-fasrc01
source activate tf2.3_cuda10

# --- Run the code ---
srun -n 1 --gres=gpu:1 python tf_mnist.py
