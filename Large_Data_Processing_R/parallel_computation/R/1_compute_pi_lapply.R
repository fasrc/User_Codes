##       author: Naeem Khoshnevis
##      created: April 2022
##      purpose: Computing pi with lapply (Monte Carlo)


source("0_helper_functions.R")

n <- 1000  # number of samples in each trial
m <- 1000000 # number of trials.

trial_vec <- (numeric(m)+1)*n

t1 <- proc.time()
pi_list_tmp <- lapply(trial_vec, monte_carlo)
t2 <- proc.time()

print(paste("Processing time: ",t2[[3]] - t1[[3]], " s."))

pi_list <- c(do.call(rbind, pi_list_tmp))
pi <- mean(pi_list)

options(digits=20)
print(paste("PI value: ", PI))
print(paste("Est.  pi: ", pi))
print(paste("Number of matched chars: ", match_chars(pi, PI)))