# Using Numba on FASRC Cluster

<img src="numba-logo.svg" alt="cuda-logo" width="200"/>

[Numba](https://numba.pydata.org) is a Python library that specializes in just-in-time (JIT) compilation of Python code for high-performance numerical computing. Developed by Anaconda Inc., Numba aims to bridge the gap between Python's ease of use and the speed of low-level languages like C and Fortran. It achieves this by automatically translating Python functions into optimized machine code, making it an invaluable tool for scientific and numerical computing, as well as data analysis.

One of the key features of Numba is its ability to accelerate code execution without the need for developers to rewrite their Python programs in a different language. Instead, Numba works seamlessly with NumPy arrays and other Python data structures, allowing users to write code that is both readable and high-performing. By adding a few decorators or function signatures, developers can instruct Numba to compile specific functions or even entire modules, resulting in significant speed improvements.

Numba supports various compilation targets, including CPU and GPU, making it versatile for a wide range of applications. It's particularly popular in the scientific and data science communities, where numerical computations often involve large datasets and complex algorithms. By leveraging Numba's JIT compilation capabilities, developers can achieve near-native performance in Python, making it a valuable tool for optimizing critical sections of code and accelerating data analysis workflows.

## Installing Numba

Create a conda environment and install `Numba` via `mamba`, e.g.,
```bash
module load python/3.10.12-fasrc01
mamba create -n numba_env python=3.10 pip wheel
source activate numba_env
mamba install numba
```

## Example: SAXPY

### SAXPY in serial

#### Source code `saxpy.py`
```python
import numpy as np
import timeit

# --- saxpy function ---
def saxpy(a, x, y):
    """
    Calculates y = a * x + y using saxpy operation.
    
    Parameters:
        a (float): Scalar constant
        x (numpy.ndarray): 1D NumPy array of size n
        y (numpy.ndarray): 1D NumPy array of size n
        
    Returns:
        y (numpy.ndarray): 1D NumPy array of size n after saxpy operation.
    """
    assert len(x) == len(y), "x and y must have the same length"
    n = len(x)    
    for i in range(n):
        y[i] = a * x[i] + y[i]        
    return y

# --- Random seed ---
np.random.seed(seed=99)

# --- Parameters ---
a = 2.0 
N = int(1e8) # Problem dimension

# --- Generate random vectors x and y ---
x = np.random.rand(N)
y = np.random.rand(N)

# --- Execute the saxpy function and time it ---
t_start = timeit.default_timer()
res = saxpy(a, x, y)
t_end = timeit.default_timer()
t = t_end - t_start

# --- Print out problem dimension and time for performing saxpy ---
print('{0} {1:d}'.format('Dimension:', N))
print('{0} {1:7.4f} {2}'.format('Time:', t, 's'))
```

#### Example batch-jobs submission script `run.sbatch`

```bash
#!/bin/bash
#SBATCH -p test
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=12000
#SBATCH -J saxpy_test
#SBATCH -o saxpy_test.out
#SBATCH -e saxpy_test.err
#SBATCH -t 30

# Load required modules
module load python/3.10.12-fasrc01
source activate numba_env

# Run the program
srun -n 1 -c 1 python saxpy.py
```
#### Example output

```
Dimension: 100000000
Time: 21.4543 s
```

### SAXPY in Numba

#### Source code `saxpy_nb.py`
```python
import numpy as np
import timeit
from numba import vectorize, float32

# --- Numba function ---
@vectorize([float32(float32, float32, float32)], target='cuda')
def saxpy(a, x, y):
    """
    Calculates y = a * x + y using saxpy operation.
    
    Parameters:
        a (float): Scalar constant
        x (numpy.ndarray): 1D NumPy array of size N
        y (numpy.ndarray): 1D NumPy array of size N
        
    Returns:
        y (numpy.ndarray): 1D NumPy array of size n after saxpy operation.
    """    
    y = a * x + y
    return y


# --- Random seed ---
np.random.seed(seed=99)

# --- Parameters ---
a = 2.0
N = int(1e8) # Problem dimension

# --- Generate random arrays x and y and convert values to float32
x = np.random.rand(N)
y = np.random.rand(N)
x = np.array(x, dtype=np.float32)
y = np.array(y, dtype=np.float32)

# --- Compute saxpy and time it ---
t_start = timeit.default_timer()
res = saxpy(a, x, y)
t_end = timeit.default_timer()
t = t_end - t_start

# --- Print out problem dimension and computing  time ---
print('{0} {1:d}'.format('Dimension:', N))
print('{0} {1:7.4f} {2}'.format('Time:', t, 's'))
```

#### Example batch-jobs submission script `run_nb.sbatch`

```bash
#!/bin/bash
#SBATCH -p gpu_test
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --mem=12000
#SBATCH -J saxpy_nb_test
#SBATCH -o saxpy_nb_test.out
#SBATCH -e saxpy_nb_test.err
#SBATCH -t 30

# Load required modules
module load python/3.10.12-fasrc01
module load cuda/12.2.0-fasrc01
source activate numba_env

# Run the program
srun -n 1 -c 1 python saxpy_np.py
```
#### Example output

```
Dimension: 100000000
Time: 0.3301 s
```
## References

* [Numba official website](https://numba.pydata.org/)
* [Numba documentation](https://numba.readthedocs.io/en/stable/index.html)