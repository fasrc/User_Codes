#!/bin/bash
#SBATCH -J image_processing
#SBATCH -o image_processing_%a.out
#SBATCH -e image_processing_%a.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 30
#SBATCH -p serial_requeue
#SBATCH --mem=4000
#SBATCH --array=1-5

# Load required software modules
source new-modules.sh
module load matlab/R2015b-fasrc01 

# Run MATLAB code
matlab -nosplash -nodesktop -nodisplay -r \
"image_process('images_in/image_in_${SLURM_ARRAY_TASK_ID}.tif','images_out/image_out_${SLURM_ARRAY_TASK_ID}.tif');exit"
