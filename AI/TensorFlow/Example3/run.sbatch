#!/bin/bash
#SBATCH -p gpu              # partition
#SBATCH -c 8                # number of cores
#SBATCH -t 00:30:00         # time in HH:MM:SS
#SBATCH -J tf_multi         # job name
#SBATCH -o tf_multi.out     # standard output file
#SBATCH -e tf_multi.err     # standard error file
#SBATCH --gres=gpu:4        # request 4 gpus
#SBATCH --mem=8G            # total memory in GB

# pull singularity image
# this is a one-time setup. Once downloaded, you don't need to pull it again
srun -c $SLURM_CPUS_PER_TASK singularity pull --disable-cache docker://tensorflow/tensorflow:latest-gpu

# --- run code tf_multi_gpu.py ---
singularity exec --nv tensorflow_latest-gpu.sif python tf_multi_gpu.py
