# Notebooks - FASRC



## Git Clone
git clone https://github.com/fasrc/User_Codes.git


## Symlink
ln -s ScratchLFS_Path Symlink_Name


## Conda Env
conda info --envs
module load Anaconda3
conda create -y -n jupyter_env python=3.6 scikit-learn matplotlib pandas numpy
source activate jupyter_env


## R Lib
getwd()
setwd()
.libPaths()
.libPaths(USER_PATH)
.libPaths( c( USER_PATH , .libPaths() ) )


install.packages("caret", dependencies = c("Depends", "Suggests"))
