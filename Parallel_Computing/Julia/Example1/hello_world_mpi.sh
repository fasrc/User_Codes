#!/bin/bash
#SBATCH -J hello_world_mpi
#SBATCH -p test  # partition
#SBATCH -n 4 # number of cores
#SBATCH --mem-per-cpu=5GB
#SBATCH -t 0-01:00 # time (D-HH-MM)
#SBATCH -e hello_world_mpi.err
#SBATCH -o hello_world_mpi.out

export UCX_WARN_UNUSED_ENV_VARS=n
module load gcc/12.2.0-fasrc01 openmpi/4.1.4-fasrc01

julia --project=~/MPIenv -e 'using MPIPreferences; MPIPreferences.use_system_binary()'
julia --project=~/MPIenv -e 'using Pkg; Pkg.build("MPI"; verbose=true)'
mpiexec -n 4 julia --project=~/MPIenv hello_world_mpi.jl