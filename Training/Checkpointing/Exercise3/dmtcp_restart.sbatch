#!/bin/bash
#SBATCH --job-name=dmtcp-restart
#SBATCH --partition=rc-testing
#SBATCH --time=00:03:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=dmtcp_restart_%j.out
#SBATCH --error=dmtcp_restart_%j.err

set -euo pipefail

module load dmtcp

cd checkpoints

if [[ ! -f dmtcp_restart_script.sh ]]; then
    echo "ERROR: dmtcp_restart_script.sh not found."
    echo "Run dmtcp_checkpoint.sbatch first."
    exit 1
fi

PORT_FILE="${SLURM_TMPDIR:-/tmp}/dmtcp_port_${SLURM_JOB_ID}"
rm -f "$PORT_FILE"

#
# Start a fresh coordinator for this new allocation.
# Continue making checkpoints every 60 seconds.
#
dmtcp_coordinator \
    --daemon \
    --port 0 \
    --port-file "$PORT_FILE" \
    --interval 60

while [[ ! -s "$PORT_FILE" ]]; do
    sleep 0.1
done

export DMTCP_COORD_HOST="$(hostname)"
export DMTCP_COORD_PORT="$(cat "$PORT_FILE")"

echo "============================================================"
echo "DMTCP restart job"
echo "============================================================"
echo "Job ID           : ${SLURM_JOB_ID}"
echo "Node             : $(hostname)"
echo "Coordinator port : ${DMTCP_COORD_PORT}"
echo "Restart          : $(date)"
echo "============================================================"

#
# Resume from the most recent successful DMTCP checkpoint.
#
./dmtcp_restart_script.sh
