# Load the R MPI package if it is not already loaded.
if (!is.loaded("mpi_initialize")) {
    library("Rmpi")
    }
 
# Tell all slaves to return a message identifying themselves
mpi.bcast.cmd( id <- mpi.comm.rank() )
mpi.bcast.cmd( ns <- mpi.comm.size() )
mpi.bcast.cmd( host <- mpi.get.processor.name() )
mpi.remote.exec(paste("I am",mpi.comm.rank(),"of",mpi.comm.size()))
 
# Test computations
x <- 5
x <- mpi.remote.exec(rnorm, x)
length(x)
print(x)
 
# Tell all slaves to close down, and exit the program
mpi.close.Rslaves(dellog = FALSE)
mpi.quit()
