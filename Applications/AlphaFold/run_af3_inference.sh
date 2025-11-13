#!/bin/bash
#SBATCH -J AF3_inference         # Job name
#SBATCH -p gpu_test              # Partition(s) (separate with commas if using multiple)
#SBATCH --gres=gpu:1             # Number of GPUs
#SBATCH -c 8                     # Number of cores
#SBATCH -t 00:10:00              # Time (HH:MM:SS)
#SBATCH --mem=8g                 # Memory
#SBATCH -o AF3_inf_%j.out        # Both stdout and stderr files

# (don't change this) set database directory (folder)
data_dir=/n/holylabs/rc_admin/Everyone/alphafold_databases/v3

# (change this) set output directory (folder)
my_output_dir=/n/holylabs/LABS/jharvard_lab/Lab/alphafold3/output_dir

# (change this) set model parameters directory (folder)
my_model_parms_dir=/n/holylabs/LABS/jharvard_lab/Lab/alphafold3/model_parameters

# note that the json_path is now output with the .json file from the data pipeline
singularity exec \
     --nv \
     --bind $data_dir:/data \
     /n/singularity_images/FAS/alphafold/alphafold_3.0.1.sif \
     python /app/alphafold/run_alphafold.py \
     --json_path=/n/holylabs/LABS/jharvard_lab/Lab/alphafold3/output_dir/2pv7/2pv7_data.json \
     --model_dir=$my_model_parms_dir \
     --output_dir=${my_output_dir}   \
     --norun_data_pipeline
