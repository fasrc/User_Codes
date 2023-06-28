# AlphaFold

**Note:** currently (Jun/2023) this documentation works for monomer simulations.
We will soon include a multimer example.

## What is AlphaFold?

[AlphaFold](https://github.com/deepmind/alphafold)

## AlphaFold in the FASRC Cannon cluster

Alphafold runs within a Docker container. However, Docker containers are not
allowed in high performance computing (HPC) systems such as Cannon because
Docker requires root/sudo privileges, which poses a security concern in HPC 
systems.

Instead, we use [Singularity
containers](https://docs.sylabs.io/guides/latest/user-guide/introduction.html)
which was specifically designed for HPC systems. We build a Singularity
container based on the Singularity definition file from
https://github.com/prehensilecode/alphafold_singularity, At the time of the
first deployment (Dec 2022), this refers to version 2.2.4 of AlphaFold. The
Singularity image is tagged with the AlphaFold version.

The AlphaFold database is stored in a cluster-wide location, meaning that
individual users **do not** have to download the AlphaFold database to run their
simulations. The database is stored in SSD as recommended by the developers.

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

### Slurm script

https://github.com/fasrc/User_Codes/blob/d0114e5ba12c2f201a1781a873420e4155572d93/Applications/AlphaFold/run_alphafold.sh#L1-L35

### Fasta file

https://github.com/fasrc/User_Codes/blob/f15ac0ea89488dc841d703fef33fa12a094779f2/Applications/AlphaFold/5ZE6_1.fasta#L1-L2

### Submitting a slurm batch job that runs AlphaFold

Log in to Cannon (see [login
instructions](https://docs.rc.fas.harvard.edu/kb/terminal-access/)). Go to the
directory where `run_alphafold.sh` is located. Then submit a slurm batch job
with the command:

```bash
sbatch run_alphafold.sh
```

This example takes about 1-2 hours to run on Cannon in the `gpu` partition with
8 cores (`-c 8`).

## Resources

* [AlphaFold GitHub](https://github.com/deepmind/alphafold)
* University of Virginia [AlphaFold docs](https://www.rc.virginia.edu/userinfo/rivanna/software/alphafold/)
* [AlphaFold discussion](https://github.com/deepmind/alphafold/issues/10) about Singularity implementation
