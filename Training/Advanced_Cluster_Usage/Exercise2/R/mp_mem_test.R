library(parallel)

# Parallel function to create a symmetric random matrix
create_symmetric_matrix_parallel <- function(n) {
  matrix <- matrix(0.0, nrow = n, ncol = n)

  # Set number of threads
  cores <- 2  # explicitly set number of threads to 2
  cat("Using", cores, "cores for parallel execution\n")

  # Split the indices for parallel execution
  indices <- split(1:n, cut(1:n, cores, labels = FALSE))

  # Define parallel task
  parallel_task <- function(rows) {
    partial_matrix <- matrix(0.0, nrow = length(rows), ncol = n)
    for (idx in seq_along(rows)) {
      i <- rows[idx]
      for (j in 1:i) {
        value <- runif(1)
        partial_matrix[idx, j] <- value
        partial_matrix[idx, j] <- value
      }
    }
    return(partial_matrix)
  }

  # Execute in parallel
  results <- mclapply(indices, parallel_task, mc.cores = cores)

  # Combine results into the matrix
  current_row <- 1
  for (partial_result in results) {
    rows_count <- nrow(partial_result)
    matrix[current_row:(current_row + rows_count - 1), ] <- partial_result
    current_row <- current_row + rows_count
  }

  # Mirror lower triangular to upper triangular part
  matrix[upper.tri(matrix)] <- t(matrix)[upper.tri(matrix)]

  return(matrix)
}

# Main function
main_parallel <- function() {
  n <- 20000
  cat("Creating a symmetric random matrix of size", n, "x", n, "in parallel...\n")

  h <- create_symmetric_matrix_parallel(n)

  cat("Hamiltonian matrix created successfully!\n")

  cat("Top-left 5x5 corner of the matrix:\n")
  print(h[1:5, 1:5])
}

# Run the parallel main function
main_parallel()

