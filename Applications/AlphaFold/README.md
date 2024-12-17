# AlphaFold

[See FASRC Docs](https://docs.rc.fas.harvard.edu/kb/alphafold/)

## AlphaFold 3

We recommend running Alphafold3 in two steps to better use cluster resources

* Step 1: Run the data pipeline on a CPU partition
* Step 2: Run inference on a GPU partition

See [slurm
partitions](https://docs.rc.fas.harvard.edu/kb/running-jobs/#Slurm_partitions)
for the specifics of each partition.

### Model parameters

See [FASRC docs](https://docs.rc.fas.harvard.edu/kb/alphafold/#Alphafold3-2).

### Data pipeline

Below you will find a slurm script example
[`run_af3_data_pipeline.sh`](run_af3_data_pipeline.sh) that uses the input file
[`alphafold3_input.json`](alphafold3_input.json) from [Alphafold3 installation
guide](https://github.com/google-deepmind/alphafold3/tree/main?tab=readme-ov-file#installation-and-running-your-first-prediction).

You will have to edit in the `run_af3_data_pipeline.sh` script:
* `SBATCH` directives to suit your needs (e.g. time `-t`, number of cores `-c`,
    amount of memory `--mem`)
* `my_model_parms_dir`: location where your model parameters are saved
* `my_input_dir`: location of `.json` input file
* `my_output_dir`: location where you would like your output to be saved

File `run_af3_data_pipeline.sh`:

```
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
```

Submit a batch job

```
sbatch run_af3_data_pipeline.sh
```

The data pipeline will produce an output file called `*_data.json` in
`my_output_dir`. In this example, the output file is called `2pv7_data.json`.

**Note 1**: If you change the number of cores (`-c`), you will also need to add the
argument `jackhmmer_n_cpu` and `nhmmer_n_cpu`. However, Alphafold3 recommends
using 8 cores:

> `--jackhmmer_n_cpu`: Number of CPUs to use for Jackhmmer. Default to min(cpu_count, 8). Going beyond 8 CPUs provides very little additional speedup.
>
> `--nhmmer_n_cpu`: Number of CPUs to use for Nhmmer. Default to min(cpu_count, 8). Going beyond 8 CPUs provides very little additionalspeedup.

**Note 2:** AlphaFold3 screen output goes to the stderr file (`.err`) rather than the
stdout file (`.out`).

### Inference

In the inference step, you will need to use the `_data.json` file that was
produced during the data pipeline step. Below you will find a slurm script
example [`run_af3_inference.sh`](run_af3_inference.sh).

You will have to edit in the `run_af3_data_pipeline.sh` script:
* `SBATCH` directives to suit your needs (e.g. time `-t`, number of cores `-c`,
    amount of memory `--mem`)
* `my_model_parms_dir`: location where your model parameters are saved
* `my_output_dir`: location where you would like your output to be saved

File `run_af3_data_pipeline.sh`:

```
#!/bin/bash
#SBATCH -J AF3_inference         # Job name
#SBATCH -p gpu_test              # Partition(s) (separate with commas if using multiple)
#SBATCH --gres=gpu:1             # Number of GPUs
#SBATCH -c 8                     # Number of cores
#SBATCH -t 00:10:00              # Time (HH:MM:SS)
#SBATCH --mem=8g                 # Memory
#SBATCH -o AF3_inf_%j.out        # Name of standard output file
#SBATCH -e AF3_inf_%j.err        # Name of standard error file

# (don't change this) set database directory
export data_dir=/n/holylfs04-ssd2/LABS/FAS/alphafold_databases/v3.0

# (change this) set output directory
export my_output_dir=/n/holylabs/LABS/jharvard_lab/Lab/alphafold3/output_dir

# (change this) set model parameters directory
export my_model_parms_dir=/n/holylabs/LABS/jharvard_lab/Lab/alphafold3/model_parameters

# note that the json_path is now output with the .json file from the data pipeline
singularity exec \
     --nv \
     --bind $data_dir:/data \
     /n/singularity_images/FAS/alphafold/alphafold_3.0.0.sif \
     python /app/alphafold/run_alphafold.py \
     --json_path=/n/holylabs/LABS/jharvard_lab/Lab/alphafold3/output_dir/2pv7/2pv7_data.json \
     --model_dir=$my_model_parms_dir \
     --output_dir=${my_output_dir}   \
     --norun_data_pipeline
```

Submit a batch job

```
sbatch run_af3_data_pipeline.sh
```

**Note:** AlphaFold3 screen output goes to the stderr file (`.err`) rather than the
stdout file (`.out`).

### File structure

For the example above, the directory structure before running Alphafold3:

```
[jharvard@boslogin07 alphafold3]$ pwd
/n/holylabs/LABS/jharvard_lab/Lab/alphafold3
[jharvard@boslogin07 alphafold3]$ tree
.
├── input_dir
│   └── alphafold3_input.json
├── model_parameters
│   └── af3.bin
└── output_dir
```

After running the data pipeline -- note the file `2pv7_data.json` in the
`output_dir`:

```
[jharvard@boslogin07 alphafold3]$ tree
.
├── input_dir
│   └── alphafold3_input.json
├── model_parameters
│   └── af3.bin
└── output_dir
    └── 2pv7
        └── 2pv7_data.json
```

After running inference:

```
[jharvard@boslogin07 alphafold3]$ tree
.
├── input_dir
│   └── alphafold3_input.json
├── model_parameters
│   └── af3.bin
└── output_dir
    └── 2pv7
        ├── 2pv7_confidences.json
        ├── 2pv7_data.json
        ├── 2pv7_model.cif
        ├── 2pv7_summary_confidences.json
        ├── ranking_scores.csv
        ├── seed-1_sample-0
        │   ├── confidences.json
        │   ├── model.cif
        │   └── summary_confidences.json
        ├── seed-1_sample-1
        │   ├── confidences.json
        │   ├── model.cif
        │   └── summary_confidences.json
        ├── seed-1_sample-2
        │   ├── confidences.json
        │   ├── model.cif
        │   └── summary_confidences.json
        ├── seed-1_sample-3
        │   ├── confidences.json
        │   ├── model.cif
        │   └── summary_confidences.json
        ├── seed-1_sample-4
        │   ├── confidences.json
        │   ├── model.cif
        │   └── summary_confidences.json
        └── TERMS_OF_USE.md
```

## AlphaFold 2

We recommend running AlphaFold2 on GPU partitions because it runs much faster
than solely using CPUs -- due to AlphaFold's GPU optimization. See [slurm
partitions](https://docs.rc.fas.harvard.edu/kb/running-jobs/#Slurm_partitions)
for the specifics of each partition.

Below you will find a slurm script example
[`run_alphafold.sh`](run_alphafold.sh) that uses the fasta file
[`5ZE6_1.fasta`](5ZE6_1.fasta).
This example assumes that `run_alphafold.sh` and `my_fasta` are located in the
same directory. If they are located in different directories, you will have to
edit `my_fasta_path`.

You will have to edit in the `run_alphafold.sh` script:
* `SBATCH` directives to suit your needs (e.g. time `-t`, number of cores `-c`, 
    amount of memory `--mem`)
* `--json_path`: file generated on the data pipeline step

**Note:** AlphaFold2 screen output goes to the stderr file (`.err`) rather than the
stdout file (`.out`).

### Monomer batch job

This example takes about 1 hour to run on Cannon in the `gpu` partition with
8 cores (`-c 8`).

Slurm script

```
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
```

Fasta file

```
>5ZE6_1
MNLEKINELTAQDMAGVNAAILEQLNSDVQLINQLGYYIVSGGGKRIRPMIAVLAARAVGYEGNAHVTIAALIEFIHTATLLHDDVVDESDMRRGKATANAAFGNAASVLVGDFIYTRAFQMMTSLGSLKVLEVMSEAVNVIAEGEVLQLMNVNDPDITEENYMRVIYSKTARLFEAAAQCSGILAGCTPEEEKGLQDYGRYLGTAFQLIDDLLDYNADGEQLGKNVGDDLNEGKPTLPLLHAMHHGTPEQAQMIRTAIEQGNGRHLLEPVLEAMNACGSLEWTRQRAEEEADKAIAALQVLPDTPWREALIGLAHIAVQRDR
```

### Multimer batch job

This example takes about 1-2 hours to run on Cannon in the `gpu` partition with
8 cores (`-c 8`).

Slurm script

```
#!/bin/bash
#SBATCH -J AF_multimer         # Job name
#SBATCH -p gpu                # Partition(s) (separate with commas if using multiple)
#SBATCH --gres=gpu:1          # number of GPUs
#SBATCH -c 8                  # Number of cores
#SBATCH -t 03:00:00           # Time (D-HH:MM:SS)
#SBATCH --mem=60G             # Memory
#SBATCH -o AF_multi_%j.out   # Name of standard output file
#SBATCH -e AF_multi_%j.err   # Name of standard error file

# set fasta file name
# NOTE: assumes this is in the directory you are running this script in
# note that you can run multiple proteins _sequentially_ (with the same model type)
# the names need to be provided as "protein1.fasta,protein2.fasta"
# if running multimer, provide one multifasta file
# indicate oligomeric state by including extra copies of a sequence
# they still require different _names_ though
my_fasta=T1083_T1084.fasta

# set fasta-specific subfolder and filepath
# handling different possible .fasta suffixes
fasta_name="${my_fasta//.fasta}"
fasta_name="${fasta_name//.faa}"
fasta_name="${fasta_name//.fa}"
mkdir -p $fasta_name
cp $my_fasta $PWD/$fasta_name
my_fasta_path=$PWD/$fasta_name/$my_fasta

# create and set path of output directory
my_output_dir=af2_out
mkdir -p $PWD/$fasta_name/$my_output_dir
my_output_dir_path=$PWD/$fasta_name/$my_output_dir

# set model type (monomer, multimer, monomer_casp14, monomer_ptm)
# see notes under fasta file if running multimer
my_model_type=multimer

# max pdb age
# use if you want to avoid recent templates
# format yyyy-mm-dd
my_max_date="2100-01-01"

# run AlphaFold multimer using Singularity
singularity run --nv --env TF_FORCE_UNIFIED_MEMORY=1,XLA_PYTHON_CLIENT_MEM_FRACTION=4.0,OPENMM_CPU_THREADS=$SLURM_CPUS_PER_TASK,LD_LIBRARY_PATH=/usr/local/cuda-11.1/targets/x86_64-linux/lib/ --bind /n/holylfs04-ssd2/LABS/FAS/alphafold_database:/data -B .:/etc --pwd /app/alphafold /n/singularity_images/FAS/alphafold/alphafold_2.3.1.sif \
--data_dir=/data/ \
--bfd_database_path=/data/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
--db_preset=full_dbs \
--fasta_paths=$my_fasta_path \
--max_template_date=$my_max_date \
--mgnify_database_path=/data/mgnify/mgy_clusters_2022_05.fa \
--model_preset=$my_model_type \
--obsolete_pdbs_path=/data/pdb_mmcif/obsolete.dat \
--output_dir=$my_output_dir_path \
--template_mmcif_dir=/data/pdb_mmcif/mmcif_files \
--uniref30_database_path=/data/uniref30/UniRef30_2021_03 \
--uniref90_database_path=/data/uniref90/uniref90.fasta \
--pdb_seqres_database_path=/data/pdb_seqres/pdb_seqres.txt \
--uniprot_database_path=/data/uniprot/uniprot.fasta \
--use_gpu_relax=True
```

Fasta file

```
>T1083
GAMGSEIEHIEEAIANAKTKADHERLVAHYEEEAKRLEKKSEEYQELAKVYKKITDVYPNIRSYMVLHYQNLTRRYKEAAEENRALAKLHHELAIVED
>T1084
MAAHKGAEHHHKAAEHHEQAAKHHHAAAEHHEKGEHEQAAHHADTAYAHHKHAEEHAAQAAKHDAEHHAPKPH
```

### Submitting a slurm batch job that runs AlphaFold

Log in to Cannon (see [login
instructions](https://docs.rc.fas.harvard.edu/kb/terminal-access/)). Go to the
directory where `run_alphafold.sh` is located. Then submit a slurm batch job
with the command:

```bash
# monomer job
sbatch run_alphafold.sh

# multimer job
sbatch run_alphafold_multi.sh
```
