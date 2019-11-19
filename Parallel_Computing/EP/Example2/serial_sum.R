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
x <- as.integer(Sys.getenv('inp'))
res <- serial_sum(x)
print(res)
