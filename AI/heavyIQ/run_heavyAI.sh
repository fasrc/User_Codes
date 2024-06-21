#!/bin/bash
#SBATCH -c 4               # number of cores 
#SBATCH -N 1               # number of nodes
#SBATCH -t 0-00:20         # time in D-HH:MM
#SBATCH -p gpu_test        # partition
#SBATCH --gres=gpu:1       # number of gpus
#SBATCH --mem-per-cpu=1G   # memory per core in GB
#SBATCH -o heavyAI.out     # standard output file
#SBATCH -e heavyAI.err     # standard error file

# 2. Set network connections
source setup_heavyAI.sh

# 3. Set working directory 
cd "${HEAVYAIBASE}"

# 4. Defines the HeavyAI singularity container to run
#export container_image="/n/singularity_images/OOD/omnisci/heavyai-ee-cuda_v7.2.2.sif"
export container_image="/n/holyscratch01/rc_admin/Lab/paulasan/singularity/heavyIQ/master_0531_iq_fix.tar.sif"

# 5. Bind slurm variables to the Singularity container
export SING_BINDS=" -B /etc/nsswitch.conf -B /etc/sssd/ -B /var/lib/sss -B /etc/slurm -B /slurm -B /var/run/munge  -B `which sbatch ` -B `which srun ` -B `which sacct ` -B `which scontrol `   -B /usr/lib64/slurm/ "

# 6. Bind HeavyAI HEAVYAI_BASE
export SING_BINDS="$SING_BINDS -B ${HEAVYAIBASE}/var/lib/heavyai:/var/lib/heavyai "

# copy files
echo "copying files"
cp /n/home_rc/paulasan/tickets/heavyIQ/*  ${HEAVYAIBASE}/var/lib/heavyai/
ls -l ${HEAVYAIBASE}/var/lib/heavyai/


# 7. Print tunneling commands
echo ""
echo "===================================================================== "
echo "execute ssh command from local computer:"
echo "ssh -NL ${port}:${SLURMD_NODENAME}:${port} ${USER}@login.rc.fas.harvard.edu"
echo "===================================================================== "
echo ""

# 8. Run HeavyAI
SINGULARITYENV_MAPD_WEB_PORT=${MAPD_WEB_PORT} SINGULARITYENV_MAPD_TCP_PORT=${MAPD_TCP_PORT} SINGULARITYENV_MAPD_HTTP_PORT=${MAPD_HTTP_PORT} SINGULARITYENV_MAPD_CALCITE_PORT=${MAPD_CALCITE_PORT} singularity run --nv $SING_BINDS --pwd /opt/heavyai $container_image


