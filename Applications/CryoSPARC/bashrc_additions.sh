# Add this section to your .bashrc on the cluster
# The following environement variables can be modified to suite your needs

# This is the location of the `cryosparc_master` directory
export CRYODIR=_______________________/cryosparc_install/cryosparc_master

# This makes sure cryosparcm is in the path
export CRYOSPARC_SBATCH=$HOME/.cryosparc_slurm_script.sh
export CRYOSPARC_CONNECTION_SCRIPT=$HOME/.cryosparc_connection_script.sh
export CRYOSPARC_OUT=$HOME/.cryosparc.out
export CRYOSPARC_ERR=$HOME/.cryosparc.err
export CRYO_WAIT_INTERVAL=5 #Seconds to wait between checking for the connection script to appear
export LOGIN=holylogin.rc.fas.harvard.edu

launchcryosparc()
{
    # Remove potentially old files
    rm $CRYOSPARC_CONNECTION_SCRIPT $CRYOSPARC_OUT $CRYOSPARC_ERR
 
    # Find an available TCP port
    for port in {7000..11000}; do ! nc -z localhost ${port} && break; done

    # Generate the slurm script
    cat  <<EOF1 > $CRYOSPARC_SBATCH
#!/bin/bash
#
#
#SBATCH -p serial_requeue # partition (queue)
#SBATCH --mem 32G # memory pool for all cores
#SBATCH -N 1 # Nodes
#SBATCH -J CryoSPARC #Job Name
#SBATCH -c 4 # Cores
#SBATCH -t 0-12:00 # time (D-HH:MM)
#SBATCH -o $CRYOSPARC_OUT
#SBATCH -e $CRYOSPARC_ERR

host=\`hostname\`

# Modify the config file with the relevant port and hostname
sed -i -E "/(CRYOSPARC_MASTER_HOSTNAME|CRYOSPARC_BASE_PORT|ROOT_URL)/d" $CRYODIR/config.sh
echo "export CRYOSPARC_MASTER_HOSTNAME=\$host" >> $CRYODIR/config.sh
echo "export CRYOSPARC_BASE_PORT=$port" >> $CRYODIR/config.sh

# Remove old dead socket files
rm /tmp/cryosparc*

# Modules needed for worker pycuda
# These should really not be hard-coded here, but
# it's okay for now. TODO
worker_modules=(
  cuda/11.1.0-fasrc01 
  cudnn/8.1.0.77_cuda11.2-fasrc01
  GCC/8.2.0-2.31.1 
  OpenMPI/3.1.3 
  Boost/1.69.0
)

#Cluster software modules go here
# Need to copy this down to the slurm job script too
module purge
module load \${cuda_modules[@]}

cryosparcm start

# This is a workaround for switching port numbers
# If you don't do this, cryosparc can't talk to
# its mongodb server. Possibly we could remove
# this is in the future. I'm not sure.
cryosparcm fixdbport
cryosparcm restart

#Purely Diagnostic
cryosparcm checkdb
cryosparcm status

tee $CRYOSPARC_CONNECTION_SCRIPT <<EOF
echo "Authenticate SSH Tunnel"
# Paste this in your terminal:
ssh -fNL $port:\$(hostname):$port $USER@$LOGIN

echo "Opening Web Browser"
# Go here:
python -m webbrowser http://localhost:$port

EOF

# Keep the slurm session alive
sleep infinity


EOF1

    # Now submit the job script
    sbatch $CRYOSPARC_SBATCH

    # Keep printing the stdout until the connection script is ready
    while [[ ! -f $CRYOSPARC_CONNECTION_SCRIPT ]];do
        echo "Waiting for CryoSPARC to start ..."
        cat $CRYOSPARC_OUT
        sleep $CRYO_WAIT_INTERVAL
        clear
    done
    echo "CryoSPARC is ready!"
    cat $CONNECTION_SCRIPT
}
