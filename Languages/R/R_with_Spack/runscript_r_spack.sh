#!/bin/bash
#SBATCH -J r_spack          # Job name
#SBATCH -c 1                # Number of cores (--cpus-per-task)
#SBATCH -t 0-00:10          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p test             # Partition to submit to
#SBATCH --mem=4g            # Memory for all cores in GB (see also --mem-per-cpu)
#SBATCH -o myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid

# source spack (i.e., make spack available for your job)
# this path is the location where you cloned spack + /share/spack/setup-env.sh
# for example: /n/holylabs/LABS/<PI_LAB>/Users/<USER_NAME>/spack/share/spack/setup-env.sh
. /n/holylabs/LABS/jharvard_lab/Users/jharvard/spack/share/spack/setup-env.sh

# print spack version
echo "spack version"
spack --version

# load spack packages
spack load r-codetools
spack load r-rgdal
spack load r-raster
spack load r-terra

# run R program and keep output and error messages in r_spack_load_libs.Rout
Rscript --vanilla r_spack_load_libs.R > r_spack_load_libs.Rout 2>&1

# run R program and keep output in r_spack_load_libs.Rout
# and error messages in standard error file
#Rscript --vanilla r_spack_load_libs.R > r_spack_load_libs.Rout
