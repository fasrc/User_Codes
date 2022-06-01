##       author: Naeem Khoshnevis
##      created: April 2022
##      purpose: Computing pi with parLapply (monte Carlo)


source("0_helper_functions.R")

library(parallel)

n <- 1000  # number of samples in each trial
m <- 100000 # number of trials.

trial_vec <- (numeric(m)+1)*n

# create cluster of workers
nthread <- 12
cl <- parallel::makeCluster(nthread, type="PSOCK")

t1 <- proc.time()
pi_list_tmp <- parallel::parLapply(cl, trial_vec, mc_pi)
t2 <- proc.time()

parallel::stopCluster(cl)

print(paste("Processing time: ",t2[[3]] - t1[[3]], " s."))

pi_list <- c(do.call(rbind, pi_list_tmp))
pi_mean <- mean(pi_list)

options(digits=20)
print(paste("PI value: ", PI))
print(paste("Est.  pi: ", pi_mean))
print(paste("Number of matched chars: ", match_chars(pi_mean, PI)))