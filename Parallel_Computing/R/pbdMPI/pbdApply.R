# This is example is a slight modification of pbdMPI's demo `pbdApply.r`

### Initial.
suppressMessages(library(pbdMPI, quietly = TRUE))
init()
.comm.size <- comm.size()
.comm.rank <- comm.rank()

### Examples.
N <- 100
x <- matrix((1:N) + N * .comm.rank, ncol = 10)
comm.print(x)

# compute sum using "master-worker" (mw) model
y <- pbdApply(x, 1, sum, pbd.mode = "mw")
comm.print(y)

# compute sum using "Single Program Multiple Data" (spmd) model
y <- pbdApply(x, 1, sum, pbd.mode = "spmd")
comm.print(y)

comm.print("This is a print statement in pbdMPI")
comm.print(.comm.size)

### Finish.
finalize()

