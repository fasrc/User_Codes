#!/bin/bash
#SBATCH -J AF3_data_pipeline     # Job name
#SBATCH -p test,serial_requeue   # Partition(s) (separate with commas if using multiple)
#SBATCH -c 8                     # Number of cores
#SBATCH -t 00:30:00              # Time (D-HH:MM:SS)
#SBATCH --mem=12g                # Memory
#SBATCH -o AF3_dp_%j.out         # Name of standard output file
#SBATCH -e AF3_dp_%j.err         # Name of standard error file

# (don't change this) set database directory
export data_dir=/n/holylfs04-ssd2/LABS/FAS/alphafold_databases/v3.0

# (change this) set model parameters directory
export my_model_parms_dir=/n/holylabs/LABS/jharvard_lab/Lab/alphafold3/model_parameters

# (change this) set input directory -- must match location of json input file
export my_input_dir=/n/holylabs/LABS/jharvard_lab/Lab/alphafold3/input_dir

# (change this) set output directory
export my_output_dir=/n/holylabs/LABS/jharvard_lab/Lab/alphafold3/output_dir

# run alphafold3
singularity exec \
     --bind $data_dir:/data \
     /n/singularity_images/FAS/alphafold/alphafold_3.0.0.sif \
     python /app/alphafold/run_alphafold.py \
     --json_path=${my_input_dir}/alphafold_input.json \
     --model_dir=$my_model_parms_dir \
     --db_dir=/data \
     --pdb_database_path=/data/mmcif_files \
     --input_dir=${my_input_dir}     \
     --output_dir=${my_output_dir}   \
     --norun_inference
