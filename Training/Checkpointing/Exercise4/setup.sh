module load gcc/15.2.0-fasrc01
module load mpich/5.0.0-fasrc01
mana_root=/n/holylabs/rc_admin/Everyone/checkpoint-training/mana
export PATH=${mana_root}/bin:$PATH
export LD_LIBRARY_PATH=${mana_root}/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=${mana_root}/lib:$LIBRARY_PATH
