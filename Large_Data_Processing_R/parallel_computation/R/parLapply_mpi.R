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

n <- 1000000  # number of samples in each trial
m <- 100000000 # number of trials.
val <- (numeric(m)+1)*n

time_a <- proc.time()
pi_list_tmp <- mpi.parLapply(val, mc_pi)
time_b <- proc.time()

print(paste("Processing time: ",time_b[[3]] - time_a[[3]], " s."))

pi_list <- c(do.call(rbind, pi_list_tmp))
pi_hat <- (mean(pi_list))

options(digits=20)
print(paste("PI value: ", PI))
print(paste("Est.  pi: ", pi_hat))
print(paste("Number of matched chars: ", match_digit(pi_hat, PI)))

# Tell all slaves to close down, and exit the program
mpi.close.Rslaves(dellog = FALSE)
mpi.quit()