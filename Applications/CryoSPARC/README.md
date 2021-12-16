Primary Author: Kevin Dalton

### What is CryoSPARC?
[CryoSPARC](https://guide.cryosparc.com/) is a closed source, commercially-developed piece software for analyzing single-particle cryoelectron microscopy data.
It supports CUDA based GPU-accelerated analysis through the PyCUDA library.
It consists of several applications which are bundled in two separate binary packages, termed
 - CryoSPARC Master (`cryosparcm`)
 - CryoSPARC Worker

The Master package is meant to use relative little compute resources, and at least some sysadmins seem to have decided to allow users to run this directly on login nodes. 
CryoSPARC Worker can be run on a separate node or the same node, but typically should have access to GPU compute resources. 
The worker nodes must have password-less SSH access to the master node as well as unfettered TCP on a number of ports. 
The authoritative list of requirements for installation can be found in the CryoSPARC [guide](https://guide.cryosparc.com/setup-configuration-and-management/cryosparc-installation-prerequisites). 

In addition to instantiating worker nodes and connecting them to the Master node, CryoSPARC can also be configured with a "Cluster Lane" which submits jobs via the SLURM job scheduler. 
This is the install strategy described in this document. 

### CryoSPARC Master
The master program is called with the `cryosparcm` command documented [here](https://guide.cryosparc.com/setup-configuration-and-management/management-and-monitoring/cryosparcm). 
The major mechanism for customizing the behavior of `cryosparcm` is the config file located in `cryosparc_master/config.sh`.
A basic `config.sh` might look like the following
```bash
export CRYOSPARC_LICENSE_ID="________-____-____-____-____________"
export CRYOSPARC_DB_PATH="_________________________________________/cryosparc_database"
export CRYOSPARC_DEVELOP=false
export CRYOSPARC_INSECURE=false
export CRYOSPARC_CLICK_WRAP=true
export CRYOSPARC_MASTER_HOSTNAME=holy_______.rc.fas.harvard.edu
export CRYOSPARC_BASE_PORT=____
```
Containing the license, path to the MongoDB datbase, master hostname, and the base tcp port. 

At the top level, `cryosparcm` is really a `Supervisor` based `shell` script which manages at least six different applications.
For instance, running `cryosparcm start` will bring up the following applications
 - app (cli)
 - command_core
 - command_rtp
 - command_vis
 - database (MongoDB)
 - webapp

As far as I can tell these mostly communicate with one another over `TCP`. 
The `TCP` ports used by each of the component programs are not individually configurable, but the base port to which the user connects is configurable in the `cryosparc_master/config.sh`. 
Of note, the hostname of the node running the master application is also typically hardcoded in this `config.sh` file. 
However, if this is left unset, it will take the hostname of the machine on which `cryosparcm start` is called. 

### Obtaining a CryoSPARC License
CryoSPARC is free for academic use. 
However, it does require a license.
You can [request a license](http://cryosparc.com/download) on the CryoSPARC webpage. 
The process of requesting a license is described in detailed [here](https://guide.cryosparc.com/setup-configuration-and-management/how-to-download-install-and-configure/obtaining-a-license-id).

### Installing CryoSPARC on Cannon
This monolithic install script is based on the [instructions](https://guide.cryosparc.com/setup-configuration-and-management/how-to-download-install-and-configure/downloading-and-installing-cryosparc) found in the CryoSPARC guide. 
This script should typically be run __on a GPU node__ so that the correct CUDA modules are loaded and functioning. 
Sequentially, this script does the following steps
 1) Set environment variables particular to this install
 2) Remove any potential old files that will cause the install to fail
 3) Load the appropriate CUDA related modules
 4) Make the installation directories
 5) [Download](https://guide.cryosparc.com/setup-configuration-and-management/how-to-download-install-and-configure/downloading-and-installing-cryosparc#use-curl-to-download-the-two-files-into-tarball-archives) the CryoSPARC Master and Worker binaries
 6) Unpack the binaries
 7) [Install](https://guide.cryosparc.com/setup-configuration-and-management/how-to-download-install-and-configure/downloading-and-installing-cryosparc#install-the-cryosparc_master-package) the master binary
 8) [Install](https://guide.cryosparc.com/setup-configuration-and-management/how-to-download-install-and-configure/downloading-and-installing-cryosparc#gpu-worker-node-cryosparc-installation) the worker binary
 9) [Add](https://guide.cryosparc.com/setup-configuration-and-management/how-to-download-install-and-configure/downloading-and-installing-cryosparc#create-the-first-user) an initial user account
 10) [Add](https://guide.cryosparc.com/setup-configuration-and-management/how-to-download-install-and-configure/downloading-and-installing-cryosparc#connect-a-cluster-to-cryosparc) the [cluster configuration](https://guide.cryosparc.com/setup-configuration-and-management/how-to-download-install-and-configure/cryosparc-cluster-integration-script-examples#slurm) for running jobs on Cannon
 
 These steps can be run sequentially or interactively if preferred. 


```bash
###################################################################################
# Configure these lines
###################################################################################

# CryoSPARC license
export LICENSE_ID="________-____-____-____-____________"

# Where to store the MongoDB database. This should be unique for each user.
# Likely this should be on Tier 1 storage at Holyoke if possible
# Use an absolute path!
export DBPATH="__________________________________"

# What slurm partition to use for CryoSPARC job submission
export SLURM_PARTITION='gpu_test' #for running jobs
export PARTITION_WALLTIME="08:00:00" 

# Where the cryosparc binaries will be installed
# This should be a fast filesystem, ideally Tier 0 storage at Holyoke
# Use an absolute path!
export INSTALL_DIR="_____________________________"

# An initial user name and password for the first login
EMAIL="_______@fas.harvard.edu"
PASSWORD="Password123"
USERNAME="____"
FIRSTNAME="____"
LASTNAME="____"

# Modules needed for CryoSPARC Worker 
worker_modules=(
    cuda/11.1.0-fasrc01 
    cudnn/8.1.0.77_cuda11.2-fasrc01
    GCC/8.2.0-2.31.1 
    OpenMPI/3.1.3 
    Boost/1.69.0
)

# Fast local node scratch 
CACHE_DIR=/scratch/cryosparc_cache

# This will be set dynamically when you launch CryoSPARC in the future
export BASE_PORT=7000

###################################################################################
# Installation commands below here
###################################################################################

# You may skip this if this is your very first install
# Remove potential old files
rm /tmp/cryosparc-supervisor-*.sock
rm -rf $INSTALL_DIR
rm -rf $CACHE_DIR
rm -rf $DBPATH

# Cluster software modules go here
module purge
module load ${cuda_modules[@]}

# Make required directories if they don't exist
mkdir -p $INSTALL_DIR
mkdir -p $CACHE_DIR
cd $INSTALL_DIR

# Download cryosparc
curl -L https://get.cryosparc.com/download/master-latest/$LICENSE_ID -o cryosparc_master.tar.gz
curl -L https://get.cryosparc.com/download/worker-latest/$LICENSE_ID -o cryosparc_worker.tar.gz

# Unpack binaries
tar xf cryosparc_master.tar.gz
tar xf cryosparc_worker.tar.gz

cd $INSTALL_DIR/cryosparc_master

# The installer no longer succeeds in modifying .bashrc
./install.sh  \
    --yes \
    --port $BASE_PORT \
    --dbpath $DBPATH \
    --license $LICENSE_ID 

# This will pull cryosparcm into the path and start it
export PATH=$INSTALL_DIR/cryosparc_master/bin:$PATH
echo "export PATH=$INSTALL_DIR/cryosparc_master/bin:\$PATH" >> ~/.bashrc
cryosparcm start 

# Install worker program
cd $INSTALL_DIR/cryosparc_worker
./install.sh  \
    --license $LICENSE_ID \
    --cudapath $CUDA_HOME \
    --yes

# Add a user account for the webapp
cryosparcm createuser \
    --email $EMAIL \
    --password $PASSWORD \
    --username $USERNAME \
    --firstname $FIRSTNAME \
    --lastname $LASTNAME

# Add "Cluster Lane"
cd $INSTALL_DIR

# Write out the cluster configuration json file
cat << EOF > cluster_info.json 
{
"qdel_cmd_tpl": "scancel {{ cluster_job_id }}",
"worker_bin_path": "$INSTALL_DIR/cryosparc_worker/bin/cryosparcw",
"title": "cannon",
"cache_path": "/scratch/tmp",
"qinfo_cmd_tpl": "sinfo --format='%.8N %.6D %.10P %.6T %.14C %.5c %.6z %.7m %.7G %.9d %20E'",
"qsub_cmd_tpl": "sbatch {{ script_path_abs }}",
"qstat_cmd_tpl": "/usr/bin/squeue -j {{ cluster_job_id }}",
"cache_quota_mb": null,
"send_cmd_tpl": "{{ command }}",
"cache_reserve_mb": 10000,
"name": "Cannon"
}
EOF

# Write out the SLURM job submission template
cat << EOF > cluster_script.sh
#!/bin/bash
#SBATCH --job-name=cryosparc_{{ project_uid }}_{{ job_uid }}
#SBATCH --partition=$SLURM_PARTITION
#SBATCH --output={{ job_log_path_abs }}
#SBATCH --error={{ job_log_path_abs }}
#SBATCH --nodes=1
#SBATCH --time=$PARTITION_WALLTIME
#SBATCH --mem={{ (ram_gb*1000)|int }}M
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={{ num_cpu }}
#SBATCH --gres=gpu:{{ num_gpu }}
#SBATCH --constraint=v100
#SBATCH --gres-flags=enforce-binding

# Load cuda modules
module purge
module load ${cuda_modules[@]}

available_devs=""
for devidx in \$(seq 1 16);
do
    if [[ -z \$(nvidia-smi -i \$devidx --query-compute-apps=pid --format=csv,noheader) ]] ; then
        if [[ -z "\$available_devs" ]] ; then
            available_devs=\$devidx
        else
            available_devs=\$available_devs,\$devidx
        fi
    fi
done
export CUDA_VISIBLE_DEVICES=\$available_devs

srun {{ run_cmd }}
EOF

cryosparcm cluster connect
```

### Running cryosparc on Cannon
To run CryoSPARC on Cannon, several things need to happen. 
First an open port on the login node in the range `7000-11000` needs to be identified (see [here](https://docs.rc.fas.harvard.edu/kb/jupyter-notebook-server-on-cluster/)). 
Next a CPU node should be allocated using the `SLURM` job scheduler. 
The CryoSPARC Master configuration file, `$INSTALL_DIR/cryosparc_master/config.sh` needs to be modified to reflect the available port and node hostname.
Then the `cryosparcm` master process can be started on the CPU node. 
In order to adapt the MongoDB database to a potentially new port number, the following commands need to be run once the Master has started. 

```shell
cryosparcm fixdbport
cryosparcm restart
```

Then the user must set up an `SSH` tunnel from their local machine through the login node to the compute node. 

```shell
ssh -NL port:compute_node:port user@holylogin.rc.fas.harvard.edu
```

Once authenticated, the CryoSPARC webapp should be viewable at `http://localhost:port`. 


### Mostly Automated CryoSPARC Connection
Because the launch process is tedious and error prone, I have automated much of it in a shell function which the user can add to their cluster `.bashrc`. 

```bash
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
```

This uses a temporary file `CRYOSPARC_CONNECTION_SCRIPT` in order to monitor the progress of the `cryosparcm` launch. 
This script is written out into the user's home directory after the server initializes. 


To connect from your local terminal, add the following to your local `.bashrc`. 

```bash
connect_to_cryosparc()
{
    LOGIN=$holylogin.rc.fas.harvard.edu
    USERNAME=______________
    CONNECTION_SCRIPT=".cryosparc_connection_script.sh"
    rsync $USERNAME@:~/$CONNECTION_SCRIPT . 
    chmod +x $CONNECTION_SCRIPT
    ./$CONNECTION_SCRIPT
}
```

Using these two shell functions, CryoSPARC is launched in two steps:
 1) On the login node, type `launchcryosparc`
 2) Once, the launch function returns, type `connect_to_cryosparc` on the local machine

