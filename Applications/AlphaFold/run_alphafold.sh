#!/bin/bash
#SBATCH -J alphafold          # Job name
#SBATCH -p gpu                # Partition(s) (separate with commas if using multiple)
#SBATCH --gres=gpu:1          # number of GPUs
#SBATCH -c 8                  # Number of cores
#SBATCH -t 03:00:00           # Time (D-HH:MM:SS)
#SBATCH --mem=60G             # Memory
#SBATCH -o alphafold_%j.out   # Name of standard output file
#SBATCH -e alphafold_%j.err   # Name of standard error file

# create and set path of output directory
my_output_dir=alphafold_output           # name of output directory
mkdir $my_output_dir                     # create output directory
my_output_dir_path=$PWD/$my_output_dir   # entire path of output directory

# set fasta file name and path
my_fasta=5ZE6_1.fasta
my_fasta_path=$PWD/$my_fasta

# run AlphaFold using Singularity
singularity run --env TF_FORCE_UNIFIED_MEMORY=1,XLA_PYTHON_CLIENT_MEM_FRACTION=4.0,OPENMM_CPU_THREADS=$SLURM_CPUS_PER_TASK,LD_LIBRARY_PATH=/usr/local/cuda-11.1/targets/x86_64-linux/lib/ -B /n/holylfs04-ssd2/LABS/FAS/alphafold_database:/data -B .:/etc --pwd /app/alphafold --nv /n/singularity_images/FAS/alphafold/alphafold_v2.2.4.sif \
    --data_dir=/data/ \
    --bfd_database_path=/data/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
    --db_preset=full_dbs \
    --fasta_paths=$my_fasta_path \
    --max_template_date=2021-07-28 \
    --mgnify_database_path=/data/mgnify/mgy_clusters_2018_12.fa \
    --model_preset=monomer \
    --obsolete_pdbs_path=/data/pdb_mmcif/obsolete.dat \
    --output_dir=$my_output_dir_path/ \
    --pdb70_database_path=/data/pdb70/pdb70 \
    --template_mmcif_dir=/data/pdb_mmcif/mmcif_files \
    --uniclust30_database_path=/data/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
    --uniref90_database_path=/data/uniref90/uniref90.fasta \
    --use_gpu_relax=True

