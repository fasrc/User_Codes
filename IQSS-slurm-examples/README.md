# IQSS-slurm-examples

### This repository includes "Hello World" scripts to run IQSS statistics packages on a slurm cluster.

* Bash
* Julia
* Matlab
* Mathematica
* Python 2
* Python 3
* R
* SAS
* Stata

#### Scripts for submitting jobs to slurm can specify slurm settings starting at the beginning of the script.
* They must come after the initial shebang (!#) line if there is one.  
* The lines should begin with #SBATCH.
* #SBATCH settings should come before any comments or they may not work.
* %x is the name of the file used to submit the job
* %j is the job id number
* when running a job array, %A is the overall job id, %a is the array index number

Typical #SBATCH lines include:<p>
`#SBATCH -n 1 # Number of cores requested; default: 1`<p>
`#SBATCH -N 1 # Ensure that all cores are on one machine; default: unset- causes poor performance`<p>
`#SBATCH -t 15 # Runtime in minutes; job killed after this amount of time; default: 10`<p>
`#SBATCH -p serial_requeue # Partition to submit to; default: serial_requeue`<p>
`#SBATCH --mem=4000 # 4GB; Memory shared across all cores in MB (see also –mem-per-cpu)`<p>
`#SBATCH --open-mode=append; default: overwrite`<p>
`#SBATCH -o %x_%j.out # Standard out goes to this file; default: slurm-JOBID.out`<p>
`#SBATCH -e %x_%j.err # Standard err goes to this file; default: slurm-JOBID.err`<p>


#### More info here: 
* https://docs.rc.fas.harvard.edu/kb/quickstart-guide/#Run_a_batch_job8230
* https://docs.rc.fas.harvard.edu/kb/running-jobs/
* https://slurm.schedmd.com/srun.html

#### More complex examples can be found here:
https://github.com/fasrc/User_Codes
