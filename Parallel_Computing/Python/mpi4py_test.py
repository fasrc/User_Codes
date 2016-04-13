#!/usr/bin/env python
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Parallel Python test program: mpi4py  
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from mpi4py import MPI

nproc = MPI.COMM_WORLD.Get_size()   # Size of communicator 
iproc = MPI.COMM_WORLD.Get_rank()   # Ranks in communicator
inode = MPI.Get_processor_name()    # Node where this MPI process runs

if iproc == 0: print "This code is a test for mpi4py."

for i in range(0,nproc):
    MPI.COMM_WORLD.Barrier()
    if iproc == i:
        print 'Rank %d out of %d' % (iproc,nproc)
        
MPI.Finalize()
