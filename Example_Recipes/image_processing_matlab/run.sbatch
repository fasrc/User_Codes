#!/bin/bash
#SBATCH -J array_test
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 20
#SBATCH -o array_test_%a.out
#SBATCH -e array_test_%a.err
#SBATCH -p serial_requeue
#SBATCH --mem=4000
#SBATCH --array=1-3
matlab -nosplash -nodesktop -nodisplay -r "video_test('test_mv_$SLURM_ARRAY_TASK_ID.mp4');exit" 2>/dev/null
