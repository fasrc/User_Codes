# AlphaFold

[See FASRC Docs](https://docs.rc.fas.harvard.edu/?post_type=epkb_post_type_1&p=27454&preview=true)

## Running AlphaFold Examples

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

https://github.com/fasrc/User_Codes/blob/d9c3f2f785a5a633617662cc84eb44788c47516f/Applications/AlphaFold/run_alphafold.sh#L1-L48

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
