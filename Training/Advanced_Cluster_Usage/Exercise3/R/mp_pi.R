#!/usr/bin/env Rscript

# Program: mp_pi.R
# Parallel Monte-Carlo PI calculation using R's parallel package

# Function to generate random numbers using LCG
lcg_rand <- function(seed) {
  a <- 69069
  c <- 1
  m <- 2147483647
  seed <- (a * seed + c) %% m
  return(list(seed = seed, value = seed / m))
}

# Function to calculate PI for a chunk of samples
calc_pi_chunk <- function(args) {
  samples <- args$samples
  seed_base <- args$seed_base
  pid <- args$pid
  
  count <- 0
  seed <- seed_base + pid * 1999
  
  for (i in 1:samples) {
    result <- lcg_rand(seed)
    seed <- result$seed
    x <- result$value
    
    result <- lcg_rand(seed)
    seed <- result$seed
    y <- result$value
    
    z <- x * x + y * y
    if (z <= 1) count <- count + 1
  }
  
  return(count)
}

# Main function
main <- function() {
  # Parse command-line arguments
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) != 2) {
    cat("Usage: Rscript mp_pi.R <number_of_samples> <number_of_processes>\n")
    quit(status = 1)
  }
  
  total_samples <- as.integer(args[1])
  nprocesses <- as.integer(args[2])
  
  # Set up parallel cluster
  library(parallel)
  cl <- makeCluster(nprocesses)
  
  # Split samples across processes
  samples_per_proc <- total_samples %/% nprocesses
  remainder <- total_samples %% nprocesses
  tasks <- list()
  base_seed <- 1202107158
  
  for (i in 1:nprocesses) {
    chunk_size <- samples_per_proc + (if (i <= remainder) 1 else 0)
    tasks[[i]] <- list(samples = chunk_size, seed_base = base_seed, pid = i - 1)
  }
  
  # Timing start
  t0 <- Sys.time()
  
  # Export functions to cluster
  clusterExport(cl, c("lcg_rand", "calc_pi_chunk"))
  
  # Run in parallel
  results <- parLapply(cl, tasks, calc_pi_chunk)
  
  # Stop cluster
  stopCluster(cl)
  
  # Combine results
  total_count <- sum(unlist(results))
  tf <- as.numeric(Sys.time() - t0, units = "secs")
  
  # Estimate PI
  pi_estimate <- 4.0 * total_count / total_samples
  
  # Output results
  cat(sprintf("Number of processes: %2d\n", nprocesses))
  cat(sprintf("Exact value of PI: %.5f\n", pi))
  cat(sprintf("Estimate of PI:    %.5f\n", pi_estimate))
  cat(sprintf("Time: %.2f sec.\n", tf))
}

# Execute main
main()
