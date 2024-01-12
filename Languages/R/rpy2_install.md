# Installing rpy2 package on Cannon

Here, we describe the steps to install the rpy2 package and use R in
Jupyter notebooks on the [OpenOnDemand VDI
portal](https://rcood.rc.fas.harvard.edu) using that package.

## Create a conda environment, install rpy2 package, and load that
   environment by selecting the corresponding kernel to start using R
   in the notebook

Create a conda environment and install the rpy2 package:

* Request a compute node:
```bash
salloc -p test -t 02:00:00 -n1 --mem 10000
```

* Load the required software module to create the conda environment:
```bash
module load python/3.10.12-fasrc01
```

* Choose a desired location to store the conda environment, e.g.,
  $HOME or your Lab directory, if you have access to it. As an
  example, we have chosen
```bash
/n/holyscratch01/rc_admin/Users/<username>
```

* Create a base conda environment with `python 3.11` at the desired
  location using the `--prefix` flag:
```bash
conda create --prefix=/n/holyscratch01/rc_admin/Users/<username>/rpy2env python=3.11 -y
```

* Activate conda environment:
```bash
source activate rpy2env
```

* Install other essential python packages, including `jupyter`:
```bash
conda install jupyter numpy matplotlib pandas scikit-learn scipy -y
```

* Install `[rpy2](https://rviews.rstudio.com/2022/05/25/calling-r-from-python-with-rpy2/)` package:
```bash
pip install rpy2
conda install r-ggplot2 
```

* Launch the Jupyter app on the OpenOnDemand VDI portal using
  [these](https://docs.rc.fas.harvard.edu/kb/virtual-desktop/)
  instructions.

You will need to set up R to point to your local R library and load
the R software module along with cmake. One can prepare a setup script
to point to the local R library that can be executed prior to
launching the Jupyter app. For instance, you can prepare a script,
`rpy2_setup.sh`, with the contents

```bash
#!/bin/bash

export R_LIBS_USER=${HOME}/apps/R/4.3.1
```
and put it in the `$HOME/apps/R` directory. Then, in the Jupyter Lab app menu, under the `Full path of script to be executed before launching jupyter (Optional)` option, enter the full path to your setup script as illustrated below:

<img src="Images/R_setup_script.png" alt="R-setup-script" width="600"/>



and load the conda environment by selecting
  the corresponding kernel:

* Set up a local R library, e.g.,
```bash
mkdir $HOME/apps/R/4.3.1
export R_LIBS_USER=$HOME/apps/R/4.3.1
```

## Use the R kernel in Jupyter Lab

* Pick up the Jupyter Lab app in the OpenOnDemand portal. Refer to [these](https://docs.rc.fas.harvard.edu/kb/virtual-desktop/) instructions.

* Set up OpenOnDemand to use the IRkernel

You will need to set up R to point to the you local R library, and also load an appropriate R software module. This can be done by preparing and executing a setup script prior launching the Jupyter app. For instance, you can prepare a script, e.g., `R_setup.sh`, with the below contents

```bash
#!/bin/bash
module load R/4.3.1-fasrc01
export R_LIBS_USER=${HOME}/apps/R/4.3.1
```
and put it in the `$HOME/apps/R` directory. Then, in the Jupyter Lab app menu, under the `Full path of script to be executed before launching jupyter (Optional)` option, enter the full path to your setup script as illustrated below:

<img src="Images/R_setup_script.png" alt="R-setup-script" width="600"/>

* Launch the Jupyter Lab app

Upon launching the Jupyter Lab app, you should see the R kernel among the available options, e.g.,

<img src="Images/R-kernel.png" alt="R-kernel" width="600"/>

* Use R in a Jupyter notebook, e.g.,

<img src="Images/R-notebook.png" alt="R-notebook" width="400"/>

## Install R Packages in Jupyter

You can also install R packages directly in a Jupyter notebook, e.g.,

<img src="Images/R-packages.png" alt="R-packages" width="600"/>