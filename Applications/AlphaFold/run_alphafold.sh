#!/bin/bash
#SBATCH -J AF_monomer         # Job name
#SBATCH -p gpu                # Partition(s) (separate with commas if using multiple)
#SBATCH --gres=gpu:1          # number of GPUs
#SBATCH -c 8                  # Number of cores
#SBATCH -t 03:00:00           # Time (D-HH:MM:SS)
#SBATCH --mem=60G             # Memory
#SBATCH -o AF_mono_%j.out   # Name of standard output file
#SBATCH -e AF_mono_%j.err   # Name of standard error file

# set fasta file name
# NOTE: assumes this is in the directory you are running this script in
# note that you can run multiple proteins _sequentially_ (with the same model type)
# the names need to be provided as "protein1.fasta,protein2.fasta"
# if running multimer, provide one multifasta file
# indicate oligomeric state by including extra copies of a sequence
# they still require different _names_ though
my_fasta=5ZE6_1.fasta

# create and set path of output directory
my_output_dir=output
mkdir -p $my_output_dir

# set model type (monomer, multimer, monomer_casp14, monomer_ptm)
# see notes under fasta file if running multimer
my_model_type=monomer

# max pdb age
# use if you want to avoid recent templates
# format yyyy-mm-dd
my_max_date="2100-01-01"

# run AlphaFold monomer using Singularity
singularity run --nv --env TF_FORCE_UNIFIED_MEMORY=1,XLA_PYTHON_CLIENT_MEM_FRACTION=4.0,OPENMM_CPU_THREADS=$SLURM_CPUS_PER_TASK,LD_LIBRARY_PATH=/usr/local/cuda-11.1/targets/x86_64-linux/lib/ --bind /n/holylfs04-ssd2/LABS/FAS/alphafold_database:/data /n/singularity_images/FAS/alphafold/alphafold_2.3.1.sif \
--data_dir=/data/ \
--bfd_database_path=/data/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
--db_preset=full_dbs \
--fasta_paths=$my_fasta \
--max_template_date=$my_max_date \
--mgnify_database_path=/data/mgnify/mgy_clusters_2022_05.fa \
--model_preset=$my_model_type \
--obsolete_pdbs_path=/data/pdb_mmcif/obsolete.dat \
--output_dir=$my_output_dir \
--pdb70_database_path=/data/pdb70/pdb70 \
--template_mmcif_dir=/data/pdb_mmcif/mmcif_files \
--uniref30_database_path=/data/uniref30/UniRef30_2021_03 \
--uniref90_database_path=/data/uniref90/uniref90.fasta \
--use_gpu_relax=True

