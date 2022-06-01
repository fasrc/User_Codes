library("data.table")
library("future")
library("future.apply")
library("tictoc")

# select one plan
#plan(sequential)    # synchronous
#plan(multisession)  # asynchronous, no forking
plan(multicore)     # asynchronous, with forking

slow_square = 
  function(x = 1) {
    x_sq = x^2 
    d = data.frame(value = x, value_squared = x_sq)
    Sys.sleep(2)
    return(d)
    }

tic()
availableCores()
future_ex = future_lapply(1:12, slow_square)
toc(log = TRUE)

