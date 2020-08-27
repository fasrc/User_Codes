#!/bin/bash
#SBATCH -n 1 # Number of job tasks (processes)
#SBATCH -N 1 # Ensure that all tasks run on one node
#SBATCH -c 4 # Number of cores per job task
#SBATCH -t 10 # Runtime in minutes
#SBATCH -p serial_requeue # Partition to submit to
#SBATCH --mem=100 # Memory per node in MB (see also --mem-per-cpu)
#SBATCH --open-mode=append
#SBATCH -o %x_%j.out # Standard out goes to this file
#SBATCH -e %x_%j.err # Standard err goes to this file
#
# Example of running a single non-interactive process through SLURM with multiple CPU cores (multi-threaded)
#
# create a script to run
SCRIPT=$(mktemp)
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
chmod u+x $SCRIPT
# jobs of more than one task and/or more than one core must be run through srun
echo "Running: srun -n $SLURM_NTASKS -c $SLURM_CPUS_PER_TASK bash $SCRIPT"
srun -n $SLURM_NTASKS -c $SLURM_CPUS_PER_TASK bash $SCRIPT
rm -f $SCRIPT

# To submit the job:
# sbatch multicore.slurm
