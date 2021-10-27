### MPI in Python: mpi4py

[mpi4py](https://mpi4py.readthedocs.io/en/stable) provides MPI bindings to the Python programming language. This document provides instructions for using mpi4py on the FAS cluster at Harvard University.

#### Installing mpi4py

We recommend the [Anaconda](https://www.anaconda.com/distribution) Python distribution. The latest version is available with the <code>python/3.8.5-fasrc01</code> software module. Since <code>mpi4py</code> is not available with the default module you need to install it in your user environment.

The most straightforward way to install <code>mpi4py</code> in your user space is to create a new [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) with the <code>mpi4py</code> package. For instance, you can do something like the below:

<pre>
module load python/3.8.5-fasrc01
conda create -n python3_env1 python numpy pip wheel mpi4py
source activate python3_env1  
</pre>

This will create a <code>conda</code> environment named <code>python3_env1</code> with the <code>mpi4py</code> package and activate it. It will also install a MPI library required by <code>mpi4py</code>. By default, the above commands will install [MPICH](https://www.mpich.org).

For most of the cases the above installation procedure should work well. However, if your workflow requires a specific flavor and/or version of MPI, you could use <code>pip</code> to install <code>mpi4py</code> in your custom conda environment as detailed below:

* Load compiler and MPI software modules:
<pre>
module load gcc/10.2.0-fasrc01
module load openmpi/4.1.1-fasrc01
</pre>

This will load [OpenMPI](https://www.open-mpi.org) in your user environment. You can also look at our [user documentation](https://docs.rc.fas.harvard.edu/kb/modules-intro) to learn more about software modules on the FAS cluster, and also search for available software modules at our [portal](https://portal.rc.fas.harvard.edu/p3/build-reports).

* Load Python (Anaconda) module:

<pre>
module load python/3.8.5-fasrc01
</pre>

* Create a conda environment:

<pre>
conda create -n python3_env2 python numpy pip wheel
</pre>

* Install <code>mpi4py</code> with <code>pip</code>:

<pre>
pip install mpi4py
</pre>

* Activate the new environment:

<pre>
source activate python3_env2
</pre>

#### Running mpi4py

Now that you have successfully installed <code>mpi4py</code> in your environment you can try some examples!

* [Example 1](Example1)
* [Example 2](Example2)

#### References

* [mpi4py official documentation](https://mpi4py.readthedocs.io/en/stable)
* [mpi4py tutorial](https://mpi4py.readthedocs.io/en/stable/tutorial.html)