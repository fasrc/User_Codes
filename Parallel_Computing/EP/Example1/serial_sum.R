#================================================
# Function: serial_sum(N)
#           Returns the sum of integers from 1
#           through N
#================================================
serial_sum <- function(x){
  k <- 0
  s <- 0
  while (k < x){
    k <- k + 1
    s <- s + k
  }
  return(s)
}

# +++ Main program +++
tid <- as.integer(Sys.getenv('SLURM_ARRAY_TASK_ID'))
res <- serial_sum(x=tid)
print(res)
