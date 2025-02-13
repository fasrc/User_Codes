# Function to create a symmetric random matrix
create_symmetric_matrix <- function(n) {
  # Initialize an n x n matrix filled with zeros
  matrix <- matrix(0.0, nrow = n, ncol = n)
  
  # Fill the matrix with random numbers and ensure symmetry
  for (i in 1:n) {
    for (j in 1:i) {  # Only fill the lower triangular part
      value <- runif(1)  # Random float between 0 and 1
      matrix[i, j] <- value
      matrix[j, i] <- value  # Symmetric element
    }
  }
  
  return(matrix)
}

# Main function
main <- function() {
  n <- 30000  # Matrix dimension
  
  cat("Creating a symmetric random matrix of size", n, "x", n, "...\n")
  
  # Create the symmetric matrix
  h <- create_symmetric_matrix(n)
  
  cat("Hamiltonian matrix created successfully!\n")
  
  # Print a small portion of the matrix (optional)
  cat("Top-left 5x5 corner of the matrix:\n")
  print(h[1:5, 1:5])
}

# Run the main function
main()
