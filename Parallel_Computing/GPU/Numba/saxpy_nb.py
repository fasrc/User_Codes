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


