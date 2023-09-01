# AlphaFold

## What is AlphaFold?

See [AlphaFold](https://github.com/deepmind/alphafold).

## AlphaFold in the FASRC Cannon cluster

Alphafold runs within a Docker container. However, Docker containers are not
allowed in high performance computing (HPC) systems such as Cannon because
Docker requires root/sudo privileges, which poses a security concern in HPC
systems.

Instead, we use [Singularity
containers](https://docs.sylabs.io/guides/latest/user-guide/introduction.html)
which was specifically designed for HPC systems.

### Singularity images

The AlphaFold singularity images are stored in a cluster-wide location, meaning
that individual users **do not** have to copy the singularity images to use
them. Singularity images are located in

```bash
/n/singularity_images/FAS/alphafold/
```

Each singularity image is tagged with the AlphaFold version

```bash
[jharvard@holylogin03 ~]$ ls -l /n/singularity_images/FAS/alphafold/
total 13G
-rwxr-xr-x. 1 root root 4.8G May 25 18:06 alphafold_2.3.1.sif
-rwxr-xr-x. 1 root root 2.9G May 25 18:10 alphafold_2.3.2.sif
-rwxr-xr-x. 1 root root 4.5G Nov  2  2022 alphafold_v2.2.4.sif
-rw-r--r--. 1 root root  733 May 25 18:13 readme.txt
```

- Version 2.3.2: Downloded from [Catguma
  DockerHub](https://hub.docker.com/layers/catgumag/alphafold/2.3.2/images/sha256-069598169a823d12a1a2c6e26d163b78abba7f59b4171c77cd5579a0636d6bd1?context=explore)
- Version 2.3.1: Downloaded from [TACC
  DockerHub](https://hub.docker.com/layers/tacc/alphafold/2.3.1/images/sha256-47a197bfc4eb36cf52e62644e5541b77fd4848e4bb9363d73e176fdb727e06d4?context=explore)
- Version 2.2.4: We build a Singularity container based on the Singularity definition file from
  https://github.com/prehensilecode/alphafold_singularity.

### AlphaFold database

The AlphaFold database is stored in a cluster-wide location, meaning that
individual users **do not** have to download the AlphaFold database to run their
simulations. The database is stored in SSD as recommended by the developers.

Database location

```bash
/n/holylfs04-ssd2/LABS/FAS/alphafold_database
```

## Running AlphaFold

We recommend running AlphaFold on GPU partitions because it runs much faster
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
* `my_fasta` to suit to your own fasta file
* (optional) `my_output_dir` if you would like your output to go somewhere else
* (optional) `my_fasta_path` 

**Note:** AlphaFold screen output goes to the stderr file (`.err`) rather than the
stdout file (`.out`).

### Monomer batch job

This example takes about 1 hour to run on Cannon in the `gpu` partition with
8 cores (`-c 8`).

Slurm script

https://github.com/fasrc/User_Codes/blob/e87e731049e121395be476064058570ed69b9f6c/Applications/AlphaFold/run_alphafold.sh#L1-L59

Fasta file

https://github.com/fasrc/User_Codes/blob/f15ac0ea89488dc841d703fef33fa12a094779f2/Applications/AlphaFold/5ZE6_1.fasta#L1-L2

### Multimer batch job

This example takes about 1-2 hours to run on Cannon in the `gpu` partition with
8 cores (`-c 8`).

Slurm script

https://github.com/fasrc/User_Codes/blob/1f69e32f23063f144c532cd892f8d5af648ad915/Applications/AlphaFold/run_alphafold_multi.sh#L1-L60

Fasta file

https://github.com/fasrc/User_Codes/blob/1f69e32f23063f144c532cd892f8d5af648ad915/Applications/AlphaFold/T1083_T1084.fasta#L1-L4

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


## Resources

* [AlphaFold GitHub](https://github.com/deepmind/alphafold)
* University of Virginia [AlphaFold docs](https://www.rc.virginia.edu/userinfo/rivanna/software/alphafold/)
* [AlphaFold discussion](https://github.com/deepmind/alphafold/issues/10) about Singularity implementation
