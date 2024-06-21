# JAX

<img src="jax_logo_250px.png" alt="JAX-logo" width="200"/>

## What is JAX?

[JAX]() is a Python library for accelerator-oriented array computation and program transformation, designed for high-performance numerical computing and large-scale machine learning.

## Installing JAX

These instructions are intended to help you install JAX on the FASRC cluster.

### GPU Support

For general information on running GPU jobs refer to our [user documentation](https://www.rc.fas.harvard.edu/resources/documentation/gpgpu-computing-on-the-cluster). To set up JAX with GPU support in your user environment, please follow the below steps:

**JAX with CUDA 12.4 and cuDNN from software modules**

These instructions set up a `conda` environment with JAX version 0.4.27, `CUDA` version 12.4 and `cuDNN` version 9.1.1.17, where `CUDA` and `cuDNN`ar loaded as software modules, `cuda/12.4.1-fasrc01` and `cudnn/9.1.1.17_cuda12-fasrc01`

```bash
# Start an interactive job on a GPU node (target the architecture where you plan to run), e.g.,
salloc -p gpu -t 0-06:00 --mem=8000 --gres=gpu:1

# Load the required modules, e.g.,
module load python/3.10.13-fasrc01
module load cuda/12.4.1-fasrc01
module load cudnn/9.1.1.17_cuda12-fasrc01

# Create a conda environment and activate it, e.g.,
mamba create -n jax0.4.27 python=3.10 pip wheel
source activate jax0.4.27

# Install JAX
mamba install jaxlib=*=*cuda* cuda-nvcc jax=0.4.27 -c conda-forge -c nvidia -y

# Install additional packages, e.g.,
mamba install pandas scikit-learn matplotlib seaborn jupyterlab ipython -y
```

**JAX with CUDA 12.4 in a conda environment**

These instructions set up a `conda` environment with `JAX` version 0.4.29 and CUDA version 12.5, where the `cuda_runtime` is installed directly in the `conda` environment. 

```bash
# Start an interactive job on a GPU node (target the architecture where you plan to run), e.g.,
salloc -p gpu -t 0-06:00 --mem=8000 --gres=gpu:1

# Load the required modules, e.g.,
module load python/3.10.13-fasrc01

# Create a conda environment and activate it, e.g.,
mamba create -n jax0.4.29 python=3.10 pip wheel
source activate jax0.4.29

# Install JAX together with all required NVIDIA GPU libraries
pip install --upgrade jax[cuda12]==0.4.29

# Install additional packages, e.g.,
mamba install pandas scikit-learn matplotlib seaborn jupyterlab ipython -y
```
**JAX as a Singularity Container**

Below instructions illustrate using a JAX Singularity container from the [NVIDIA NGC catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/jax):

```bash
# Pull a JAX container with singularity, e.g,
singularity pull docker://nvcr.io/nvidia/jax:23.10-paxml-py3

# Example usage on the "kempner_h100" partition (H100s nodes)
$ salloc -p kempner_h100 -t 0-06:00 --mem=8000 --gres=gpu:4

$ singularity exec --nv jax_23.10-paxml-py3.sif ipython
Python 3.10.12 (main, Jun 11 2023, 05:26:28) [GCC 11.4.0]
Type 'copyright', 'credits' or 'license' for more information
IPython 8.16.1 -- An enhanced Interactive Python. Type '?' for help.

In [1]: import jax

In [2]: jax.default_backend()
Out[2]: 'gpu'

In [3]: jax.devices()
Out[3]: [cuda(id=0), cuda(id=1), cuda(id=2), cuda(id=3)]

In [4]: key = random.key(0)
...: x = random.normal(key, (10,))
...: print(x)

In [5]: import jax.numpy as jnp
...: from jax import grad, jit, vmap
...: from jax import random

In [6]: key = random.key(0)
...: x = random.normal(key, (10,))
...: print(x)
[-0.3721109 0.26423115 -0.18252768 -0.7368197 -0.44030377 -0.1521442
-0.67135346 -0.5908641 0.73168886 0.5673026 ]

In [7]: size = 3000
...: x = random.normal(key, (size, size), dtype=jnp.float32)
...: %timeit jnp.dot(x, x.T).block_until_ready() # runs on the GPU
528 µs ± 72.8 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)

In [8]: import numpy as np
...: x = np.random.normal(size=(size, size)).astype(np.float32)
...: %timeit jnp.dot(x, x.T).block_until_ready()
7.62 ms ± 12 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

In [9]: from jax import device_put
...:
...: x = np.random.normal(size=(size, size)).astype(np.float32)
...: x = device_put(x)
...: %timeit jnp.dot(x, x.T).block_until_ready()
493 µs ± 2.45 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

In [10]:
```

### Examples

Check back soon!

### Resources

* [JAX documentation](https://jax.readthedocs.io/en/latest/index.html)
* [JAX tutorials](https://jax.readthedocs.io/en/latest/tutorials.html)
