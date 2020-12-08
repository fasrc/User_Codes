#!/bin/bash
#SBATCH -n 4 # Number of job tasks (processes)
#SBATCH -N 2 # Ensure that the tasks are distributed across at least this many nodes
#SBATCH -c 2 # Number of cores per job task
#SBATCH -t 10 # Runtime in minutes
#SBATCH -p test # Partition to submit to
#SBATCH --mem=100 # Memory per node in MB (see also --mem-per-cpu)
#SBATCH --open-mode=append
#SBATCH -o %x_%j.out # Standard out goes to this file
#SBATCH -e %x_%j.err # Standard err goes to this file
#
# Example of running multiple simultaneous instances of a non-interactive process on multiple nodes through SLURM 
#  Note: this runs them all simultaneously (necessary if they need to communicate, e.g. for MPI).
#  If the processes don't need to communicate, it is better to run them asynchonously as a Job Array.
#
# create a script to run
GROUP='iqss_lab'
if [ -z "$SCRATCH" ]; then
  echo "$0: ERROR: Cannot run a multi-node script without a shared \$SCRATCH" >&2
  exit 1
fi
SCRIPT=$(mktemp -p $SCRATCH/$GROUP)
if [ -z "$SCRIPT" ]; then
  echo "$0: ERROR: Could not create tempfile" >&2
  exit 1
fi
cat > $SCRIPT << EOF
# note that all the dollar signs and backslashes are escaped because this is a heredoc
print_info() {
  cpuset=\$(cat /sys/fs/cgroup/cpuset/slurm/uid_\$(id -u)/job_\${SLURM_JOB_ID}/cpuset.cpus)
  memory_limit=\$(cat /sys/fs/cgroup/memory/slurm/uid_\$(id -u)/job_\${SLURM_JOB_ID}/memory.limit_in_bytes)
  echo "PID \$\$ running on \$(hostname) allocated cpuset \\"\$cpuset\\" memory limit \\"\$memory_limit\\""
}
print_info
EOF
# jobs of more than one task/core/node must be run through srun
echo "Running: srun -n $SLURM_NTASKS -c $SLURM_CPUS_PER_TASK -N $SLURM_JOB_NUM_NODES bash $SCRIPT"
srun -n $SLURM_NTASKS -c $SLURM_CPUS_PER_TASK -N $SLURM_JOB_NUM_NODES bash $SCRIPT
rm -f $SCRIPT

# To submit the job:
# sbatch multiprocess.slurm
