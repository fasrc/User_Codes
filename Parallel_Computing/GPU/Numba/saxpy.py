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

