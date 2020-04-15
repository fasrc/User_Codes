#!/usr/bin/env python3
from scipy.optimize import minimize
import numpy as np
from mpi4py import MPI

# Function to minimize
def f(x):
    r = x*x
    return r

# Main program
if __name__ == '__main__':
    comm  = MPI.COMM_WORLD
    nproc = comm.Get_size() 
    iproc = comm.Get_rank()

    # Define initial data on the root MPI process
    if iproc == 0:
        data = [(i+1)**2 for i in np.arange(nproc)]
    else:
        data = None

    # Scatter data to all MPI processes
    x0 = comm.scatter(data, root=0)

    # Minimize the function
    comm.Barrier()
    res = minimize(f, x0, method='Nelder-Mead', tol=1e-8)
    x = res.x[0]

    # Gather results on the root process
    x1 = comm.gather(x, root=0)
    if iproc == 0:
        print (x1)
    
