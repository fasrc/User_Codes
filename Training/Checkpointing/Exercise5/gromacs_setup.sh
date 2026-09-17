module load gcc/15.2.0-fasrc01 openmpi/5.0.10-fasrc01
export gromacs_home=/n/holylabs/rc_admin/Everyone/checkpoint-training/gromacs/sw
export PATH=${gromacs_home}/bin:$PATH
export LD_LIBRARY_PATH=${gromacs_home}/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=${gromacs_home}/lib64:$LIBRARY_PATH
export CPATH=${gromacs_home}/include:$CPATH
export FPATH=${gromacs_home}/include:$FPATH
