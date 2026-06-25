#!/bin/bash
#SBATCH -J compute_pi
#SBATCH -p test  # partition
#SBATCH -n 4 # number of cores
#SBATCH --mem-per-cpu=5GB
#SBATCH -t 0-01:00 # time (D-HH-MM)
#SBATCH -e compute_pi.err
#SBATCH -o compute_pi.out

export UCX_WARN_UNUSED_ENV_VARS=n
module load gcc/12.2.0-fasrc01 openmpi/4.1.4-fasrc01

julia --project=~/MPIenv -e 'using MPIPreferences; MPIPreferences.use_system_binary()'
julia --project=~/MPIenv -e 'using Pkg; Pkg.build("MPI"; verbose=true)'
mpiexec -n 4 julia --project=~/MPIenv compute_pi.jl