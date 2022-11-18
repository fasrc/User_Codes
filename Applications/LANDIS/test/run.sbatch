#!/bin/bash
#SBATCH -J landis_test
#SBATCH -o landis_test.out
#SBATCH -e landis_test.err
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 0-00:30
#SBATCH -p test
#SBATCH --mem=4000

# Setup
export LANDIS_ROOT=/n/home06/pkrastev/src/LANDIS/2022
export LANDIS_CONSOLE=${LANDIS_ROOT}/vol2/Core-Model-v7-LINUX/build/Release/Landis.Console.dll

# Run the program
srun -c 1 singularity exec ${LANDIS_ROOT}/Container/Singularity.simg dotnet $LANDIS_CONSOLE scenario_s1e1.txt
