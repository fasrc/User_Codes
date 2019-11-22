# Load the R MPI package if it is not already loaded.
if (!is.loaded("mpi_initialize")) {
    library("Rmpi")
    }
 
#
# In case R exits unexpectedly, have it automatically clean up
# resources taken up by Rmpi (slaves, memory, etc...)
.Last <- function(){
       if (is.loaded("mpi_initialize")){
           if (mpi.comm.size(1) > 0){
               print("Please use mpi.close.Rslaves() to close slaves.")
               mpi.close.Rslaves()
           }
           print("Please use mpi.quit() to quit R")
           .Call("mpi_finalize")
       }
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
