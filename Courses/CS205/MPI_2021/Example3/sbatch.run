#!/bin/bash -l
#SBATCH -J mpi_pi
#SBATCH -o mpi_pi.out
#SBATCH -e mpi_pi.err
#SBATCH -t 0-00:30
#SBATCH -p test
#SBATCH -n 16
#SBATCH -N 1
#SBATCH --mem-per-cpu=1000

PRO=mpi_pi
rm -rf ${PRO}.dat
touch ${PRO}.dat

# Load required software modules
module load gcc/10.2.0-fasrc01 openmpi/4.1.1-fasrc01

# Run program
for i in 1 2 4 8 16
do
    echo "Number of processes: ${i}" >> ${PRO}.dat
    srun -n ${i} --mpi=pmix ./${PRO}.x 1000000000 >> ${PRO}.dat
    echo " "
done

cat mpi_pi.dat  | grep -e Number -e time | awk '{if (NR%2 == 1 ){printf "%s ", $4}else{print $4}}'  > scaling_results.txt

sleep 2
module load python/3.8.5-fasrc01
python speedup.py
