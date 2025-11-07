import random

def create_symmetric_matrix(n):
    """
    Creates a symmetric random matrix of size n x n using pure Python.
    """
    # Initialize an n x n matrix filled with zeros
    matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    
    # Fill the matrix with random numbers and ensure symmetry
    for i in range(n):
        for j in range(i + 1):  # Only fill the lower triangular part
            value = random.random()  # Random float between 0.0 and 1.0
            matrix[i][j] = value
            matrix[j][i] = value  # Symmetric element
    
    return matrix

def main():
    n = 30000  # Matrix dimension
    
    print(f"Creating a symmetric random matrix of size {n}x{n}...")
    
    # Create the symmetric matrix
    h = create_symmetric_matrix(n)
    
    print("Hamiltonian matrix created successfully!")
    
    # Print a small portion of the matrix (optional)
    print("Top-left 5x5 corner of the matrix:")
    for row in h[:5]:
        print(row[:5])

if __name__ == "__main__":
    main()

