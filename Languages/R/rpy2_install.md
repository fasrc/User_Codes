# Installing rpy2 package on Cannon

Here, we describe the steps to install the rpy2 package and use R in
Jupyter notebooks on the [OpenOnDemand VDI
portal](https://rcood.rc.fas.harvard.edu) using that package.

## Create a conda environment, install rpy2 package, and load that
   environment by selecting the corresponding kernel to start using R
   in the notebook

Complete the below steps to create a conda environment and install the
rpy2 package:

* Request a compute node:
```bash
salloc -p test -t 02:00:00 -n1 --mem 10000
```

* Load the required software module to create the conda environment:
```bash
module load python/3.10.12-fasrc01
```

* Set up a local R library, e.g.,
```bash
mkdir $HOME/apps/R/4.3.1
export R_LIBS_USER=$HOME/apps/R/4.3.1
```

* Install `IRkernel` in R, e.g.,
```R
# --- Install IRkernel in R ---
install.packages("IRkernel")
# --- Run IRkernel in R ---
IRkernel::installspec()
```

* Copy the `IRkernel` directory to your `Jupyter` data directory. For example, if your `IRkernel` directory is in `$HOME/apps/R/4.3.1/IRkernel`, do:

```bash
cp -R $HOME/apps/R/4.3.1/IRkernel $HOME/.local/share/jupyter/
```

* Change the R path (the first "R") in the `$HOME/.local/share/jupyter/IRkernel/kernelspec /kernel.json` file to, e.g., 

```bash
/n/sw/helmod-rocky8/apps/Core/R/4.3.1-fasrc01/bin/R
```
or other `R`. For instance, if you use `R` that comes with the `R/4.3.1-fasrc01` software module, the `kernel.json` should look like the below:

```json 
{"argv": ["/n/sw/helmod-rocky8/apps/Core/R/4.3.1-fasrc01/bin/R", "--slave", "-e", "IRkernel::main()", "--args", "{connection_file}"],
 "display_name":"R",
 "language":"R"
}
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