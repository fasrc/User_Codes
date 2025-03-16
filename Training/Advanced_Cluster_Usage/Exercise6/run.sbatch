#!/bin/bash
#SBATCH -J omp_pi
#SBATCH -o omp_pi.out
#SBATCH -e omp_pi.err
#SBATCH -t 0-00:30
#SBATCH -p test
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=4G

PRO=omp_pi
rm -rf ${PRO}.dat speedup.png
touch ${PRO}.dat

# --- Load required software modules ---
module load intel/24.2.1-fasrc01
unset OMP_NUM_THREADS
# --- Run program with 1, 2, 4, and 8 OpenMP threads ---
for i in 1 2 4 8 
do
    echo "Number of threads: ${i}"
    ./${PRO}.x 500000000 ${i} >> ${PRO}.dat
    echo " "
done

# --- Generate "scaling_results.txt" data file ---
cat omp_pi.dat  | grep -e Time  -e Number | awk -F ":" '{if ($1 ~ "Time" ) {print $2}else{printf "%d ",$2}}' | awk '{print $1,$2}' > scaling_results.txt

#  --- Generate speedup figure ---
sleep 2
module load python/3.10.13-fasrc01
source activate python-3.10_env
python speedup.py
