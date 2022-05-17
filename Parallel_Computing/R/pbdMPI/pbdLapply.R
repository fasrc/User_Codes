# This is example is a slight modification of pbdMPI's demo `pbdLApply.r`

# Initial
library(pbdMPI, quietly = TRUE)
init()
.comm.size <- comm.size()
.comm.rank <- comm.rank()

# Examples
N <- 100
x <- split((1:N) + N * .comm.rank, rep(1:10, each = 10))

comm.print("Master-worker model")
y <- pbdLapply(x, sum, pbd.mode = "mw")
comm.print(unlist(y))

comm.print("Single program multiple data")
y <- pbdLapply(x, sum, pbd.mode = "spmd")
comm.print(unlist(y))

comm.print("Distributed model")
y <- pbdLapply(x, sum, pbd.mode = "dist")
comm.print(unlist(y))

# Finish
finalize()

