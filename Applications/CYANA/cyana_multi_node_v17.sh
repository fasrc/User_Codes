#!/bin/bash
#SBATCH -J cyana_multi_node    # job name
#SBATCH -n 64                  # MPI tasks (start small for testing; CYANA = 1 master + N-1 workers)
##SBATCH -N 2                   # 2 nodes for the multi-node correctness test
#SBATCH -p dsouza              # partition (use one that provides the cores you need)
#SBATCH -o %J.out
#SBATCH -e %J.err
#SBATCH -t 0:30:00
#SBATCH --mem-per-cpu=3G

# print all commands
set -x

# singularity image with cyana 3.98 (Intel MPI build, v13 with PMIx client)
sing_img=/n/netscratch/dsouza_lab/Lab/singularity_images/cyana_intel_impi_v17.sif

# No host modules needed: the container carries the Intel toolchain, Intel MPI,
# and a PMIx client. Slurm on the host provides the PMIx launch service.

sub_dir=HIV27-2
input=calc.cya

# ---------------------------------------------------------------------------
# MPI / PMI / fabric wiring for Intel MPI launched by `srun --mpi=pmix`
# ---------------------------------------------------------------------------
# Use the PMIx 5.0.1 client built into the image (/opt/pmix/lib/libpmix.so).
# It connects to the host Slurm PMIx server over a socket in the tmp dir, which
# Singularity bind-mounts - so, unlike PMI-2, nothing extra needs binding.
PMIX_LIB=/opt/pmix/lib/libpmix.so

# PMIx on the host authenticates with MUNGE (psec:munge), so the container
# needs the host munge daemon socket bind-mounted in. Path is standard on
# RHEL/Rocky; verify with `ls -l /var/run/munge/munge.socket.2` on a node.
export SINGULARITY_BIND=/var/run/munge/munge.socket.2

# Fabric: start correctness-first over TCP. Once the 2-node run is clean, switch
# FI_PROVIDER to your InfiniBand provider (mlx if UCX present, else verbs).
export I_MPI_FABRICS=shm:ofi
export FI_PROVIDER=verbs

# PMIx safeguard recommended by FASRC (also covers PMIx 5.0.0/5.0.1 skew).
export PMIX_MCA_gds=hash

# Environment variables that must reach the ranks INSIDE the container.
export SINGULARITYENV_CYANALIB=/opt/cyana-3.98.15
export SINGULARITYENV_CYANAARG=${input%.cya}
export SINGULARITYENV_I_MPI_PMI_LIBRARY=${PMIX_LIB}
export SINGULARITYENV_I_MPI_PMI=pmix
export SINGULARITYENV_I_MPI_FABRICS=${I_MPI_FABRICS}
export SINGULARITYENV_FI_PROVIDER=${FI_PROVIDER}
export SINGULARITYENV_PMIX_MCA_gds=${PMIX_MCA_gds}

# Debugging (remove once it works): confirms "PMI API: pmix" and the rank/size.
#export SINGULARITYENV_I_MPI_DEBUG=6

## Setting up scratch ##
#STORAGE_DIR="/n/netscratch/rc_admin/Lab/paulasan/cyana/batch/${SLURM_JOB_ID}.${sub_dir}"
STORAGE_DIR="/n/netscratch/dsouza_lab/Lab/$USER/${SLURM_JOB_ID}.${sub_dir}"
export STORAGE_DIR
mkdir -pv $STORAGE_DIR
echo -n ">>>>> Changing to $STORAGE_DIR: "
cd $STORAGE_DIR
pwd

##  Copying relevant files  ##
#cp -a /n/netscratch/rc_admin/Lab/paulasan/cyana/batch/demo/${sub_dir}/* ${STORAGE_DIR}
cp -a $SLURM_SUBMIT_DIR/$sub_dir/* $STORAGE_DIR
ls -l
echo "========== START OF ${input} =========="
cat ${input}
echo "========== END OF ${input} =========="

##  Executing CYANA across the allocation via Slurm PMIx ##
echo "Executing cyana..."
time srun -n $SLURM_NTASKS --mpi=pmix \
    singularity exec ${sing_img} \
    /opt/cyana-3.98.15/cyanaexe.ifx-openmpi > ${input}.out

echo "VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV"
echo -n ">> Job finished @ "
date
echo ">> Output can be found in: $SLURM_SUBMIT_DIR"
