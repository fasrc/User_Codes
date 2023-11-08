#!/bin/bash
#SBATCH -p gpu
#SBATCH -c 8
#SBATCH -t 00:30:00
#SBATCH -J tf_test
#SBATCH -o tf_test.out
#SBATCH -e tf_test.err
#SBATCH --gres=gpu:4
#SBATCH --mem=8G

# pull singularity image
# this is a one-time setup. Once downloaded, you don't need to pull it again
srun -c $SLURM_CPUS_PER_TASK singularity pull --disable-cache docker://tensorflow/tensorflow:latest-gpu

# --- run code tf_test_multi_gpu.py ---
singularity exec --nv tensorflow_latest-gpu.sif python tf_test_multi_gpu.py
