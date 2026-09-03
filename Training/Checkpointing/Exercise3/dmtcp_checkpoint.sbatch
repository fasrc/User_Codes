#!/bin/bash
#SBATCH --job-name=dmtcp-ckpt
#SBATCH --partition=rc-testing
#SBATCH --time=00:03:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=dmtcp_ckpt_%j.out
#SBATCH --error=dmtcp_ckpt_%j.err

set -euo pipefail

module load dmtcp

mkdir -p checkpoints
cd checkpoints

#
# Use a random coordinator port to avoid collisions with other users.
#
PORT_FILE="${SLURM_TMPDIR:-/tmp}/dmtcp_port_${SLURM_JOB_ID}"
rm -f "$PORT_FILE"

#
# Start a DMTCP coordinator and checkpoint once every 60 seconds.
#
dmtcp_coordinator \
    --daemon \
    --port 0 \
    --port-file "$PORT_FILE" \
    --interval 60

#
# Wait for the coordinator to write its port number.
#
while [[ ! -s "$PORT_FILE" ]]; do
    sleep 0.1
done

export DMTCP_COORD_HOST="$(hostname)"
export DMTCP_COORD_PORT="$(cat "$PORT_FILE")"

echo "============================================================"
echo "DMTCP checkpoint job"
echo "============================================================"
echo "Job ID           : ${SLURM_JOB_ID}"
echo "Node             : $(hostname)"
echo "Coordinator port : ${DMTCP_COORD_PORT}"
echo "Start            : $(date)"
echo "============================================================"

#
# The application itself contains NO checkpointing code.
#
dmtcp_launch \
    --join-coordinator \
    ../long_run.x \
    --iterations 600 \
    --sleep-ms 1000
