## Matlab

<img src="Images/matlab-logo.png" alt="R-logo" width="400"/>

### Introduction

[MATLAB](https://www.mathworks.com/products/matlab.html), short for Matrix Laboratory, is a high-performance language and interactive environment for numerical computation, visualization, and programming. Developed by MathWorks, MATLAB integrates computation, visualization, and programming in an easy-to-use environment where problems and solutions are expressed in familiar mathematical notation. It is widely used in academia and industry for data analysis, algorithm development, and modeling and simulation. One of MATLAB's key strengths is its extensive library of built-in functions and toolboxes, which cover a wide range of scientific and engineering applications, including signal processing, control systems, neural networks, and machine learning.

In addition to its robust built-in capabilities, MATLAB offers a user-friendly interface that simplifies the process of writing and debugging code. The platform supports both procedural and object-oriented programming, providing flexibility in how users approach problem-solving. MATLAB’s powerful plotting functions make it easy to visualize data and results, which is crucial for interpreting complex numerical information. Furthermore, MATLAB supports integration with other programming languages like C, C++, Java, and Python, allowing for versatile application development. Its widespread adoption in various fields, from finance and biotechnology to automotive and aerospace engineering, underscores its versatility and effectiveness as a tool for both research and practical problem-solving.

### MATLAB at FAS Research Computing

FAS Research Computing has licensed MATLAB from [Mathworks](https://www.mathworks.com/) for use on desktops, laptops, and on the FASRC cluster. If you wish to run MATLAB on your desktop/laptop, please follow the instructions on the [FAS downloads page](https://huit.harvard.edu/fas-software-downloads) (downloads for all platforms are available from Mathworks). Running MATLAB on the cluster can be done through a GUI using VDI, at the command line interactive, or through batch jobs. Below we will discuss each of these modes in details.

>**NOTE!** These instructions discuss the single-core (process) implementation. If you wish to run MATLAB as a multi-core/process job, please see our [companion document](https://docs.rc.fas.harvard.edu/kb/parallel-matlab-pct-dcs/).

### MATLAB GUI on the FASRC cluster

The MATLAB GUI can be run using the power of the compute nodes of the cluster by initiating your session via our graphical login, and starting an interactive MATLAB session. This is almost like running MATLAB on your desktop/laptop, except all the computation is done on the cluster. There two ways of using the MATLAB GUI on the cluster:

#### MATLAB VDI APP

* Log on to the cluster via [rcood.rc.fas.harvard.edu](rcood.rc.fas.harvard.edu/). For more details on Open OnDemand (VDI) go [here](https://docs.rc.fas.harvard.edu/kb/virtual-desktop/).
* From the dashboard select the `Matlab` app.
* Specify the resources for your job (e.g., partition, total memory, number of cores, Matlab version, etc.)
* Launch the job.

#### MATLAB software module 

* Log on to the cluster via [rcood.rc.fas.harvard.edu](rcood.rc.fas.harvard.edu/). 
* From the dashboard select the `Remote Desktop` app.
* Specify the resources for your job and launch it.
* Once the job start, open a `terminal` in the remote desktop.
* In the terminal, load the MATLAB module
```bash
module load matlab # This will load the default (latest MATLAB version)
```
* Start the MATLAB GUI. In the terminal type in
```bash
matlab
```

### MATLAB on the command line (interactive terminal session)

MATLAB can also be run on the command line in an interactive terminal session. Since there is no GUI, you must include additional parameters (`-nojvm -nosplash -nodesktop`), and you can optionally specify an M file (e.g., `script.m` or `function.m`) to run, including any script or function parameters as required:

* Login to the cluster as explained [here](https://docs.rc.fas.harvard.edu/kb/access-and-login/).
* Once logged in, get an interactive session as described [here](https://docs.rc.fas.harvard.edu/kb/running-jobs/#Interactive_jobs_and_srun), e.g.,
```bash
salloc --mem=4G -t 60 -c 1 -N 1 -p test
```
* Load a Matlab module, e.g,
```bash
module load matlab/R2022b-fasrc01
```
* Start MATLAB interactively without a GUI
```bash
matlab -nosplash -nodesktop -nodisplay
```

### MATLAB batch jobs

MATLAB can also be run through batch jobs. Below are the steps to accomplish this:

* Login to the cluster.
* Prepare a batch-job submission script, e.g.,
```bash
#!/bin/bash
#SBATCH -J pi_monte_carlo       # job name
#SBATCH -o pi_monte_carlo.out   # standard output file
#SBATCH -e pi_monte_carlo.err   # standard error file
#SBATCH -p test                 # partition
#SBATCH -c 1                    # number of cores
#SBATCH -t 0-00:30              # time in D-HH:MM
#SBATCH --mem=4000              # memory in MB

# Load required software modules
module load matlab
srun -c $SLURM_CPUS_PER_TASK matlab -nosplash -nodesktop -nodisplay -r "pi_monte_carlo"
```
NOTE: This assumes that you have a MATLAB script named `pi_monte_carlo.m`.
* Submit the job to the queue. If the batch-job submission script is named, e.g., `run.sbatch`, the job is sent to the queue with
```bash
sbatch run.sbatch
```

### Examples

To get started with Matlab on the FAS cluster you can try the below examples:

* [Example1](https://github.com/fasrc/User_Codes/tree/master/Languages/MATLAB/Example1): Monte-Carlo computation of PI
* [Example2](https://github.com/fasrc/User_Codes/tree/master/Languages/MATLAB/Example2): Sums up integers from 1 to N 
* [Example3](https://github.com/fasrc/User_Codes/tree/master/Languages/MATLAB/Example3): Generates a multi-figure on a 3 X 3 grid
* [Example4](https://github.com/fasrc/User_Codes/tree/master/Languages/MATLAB/Example4): Illustration of job arrays and parameter sweeps
* [Example5](https://github.com/fasrc/User_Codes/tree/master/Languages/MATLAB/Example5): Random vectors and job arrays

### References:

* [Official Matlab Documentation](https://www.mathworks.com/help/matlab)
* [Official Matlab Tutorials](https://www.mathworks.com/support/learn-with-matlab-tutorials.html)
* [Matlab Tutorial from Tutorialspoint](https://www.tutorialspoint.com/matlab/index.htm)