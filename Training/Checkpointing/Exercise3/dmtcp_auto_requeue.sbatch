#!/bin/bash
#SBATCH --job-name=dmtcp-auto
#SBATCH --partition=rc-testing
#SBATCH --time=00:03:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=dmtcp_auto_%j.out
#SBATCH --error=dmtcp_auto_%j.err
#SBATCH --signal=B:USR1@60
#SBATCH --requeue
#SBATCH --open-mode=append

set -euo pipefail

module load dmtcp

mkdir -p checkpoints
cd checkpoints

PORT_FILE="${SLURM_TMPDIR:-/tmp}/dmtcp_port_${SLURM_JOB_ID}"
rm -f "$PORT_FILE"

#
# Start a new DMTCP coordinator for this allocation.
#
# Periodic checkpoints remain useful as protection against
# interruptions that occur without advance warning.
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
echo "DMTCP automatic checkpoint/restart"
echo "============================================================"
echo "Job ID        : ${SLURM_JOB_ID}"
echo "Restart count : ${SLURM_RESTART_COUNT:-0}"
echo "Node          : $(hostname)"
echo "Time          : $(date)"
echo "============================================================"


#
# Called when Slurm sends USR1 before the wall-time limit.
#
checkpoint_and_requeue()
{
    echo
    echo "USR1 received at $(date)"
    echo "Creating final checkpoint..."

    dmtcp_command --checkpoint

    echo "Checkpoint complete."
    echo "Requeueing job ${SLURM_JOB_ID}..."

    scontrol requeue "${SLURM_JOB_ID}"

    exit 0
}

trap checkpoint_and_requeue USR1


#
# First execution:
#     launch application normally under DMTCP.
#
# Restarted execution:
#     restore from the most recent successful checkpoint.
#
if [[ ${SLURM_RESTART_COUNT:-0} -eq 0 ]]; then

    echo "Starting new DMTCP computation."

    dmtcp_launch \
        --join-coordinator \
        ../long_run.x \
        --iterations 600 \
        --sleep-ms 1000 &

else

    if [[ ! -f dmtcp_restart_script.sh ]]; then
        echo "ERROR: restart requested but no checkpoint exists."
        exit 1
    fi

    echo "Restarting DMTCP computation."

    ./dmtcp_restart_script.sh &

fi

APP_PID=$!


#
# Keep the batch shell alive so it can receive USR1.
#
wait "$APP_PID"

status=$?

echo
echo "Application finished with status ${status}."
echo "End: $(date)"

exit "$status"
