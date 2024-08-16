#!/bin/bash

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
