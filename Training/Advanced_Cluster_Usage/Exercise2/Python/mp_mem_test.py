import random
import threading

def create_symmetric_matrix_parallel(n, num_threads=2):
    """
    Creates a symmetric random matrix of size n x n using Python threading.
    """
    matrix = [[0.0 for _ in range(n)] for _ in range(n)]

    def worker(start_row, end_row):
        for i in range(start_row, end_row):
            for j in range(i + 1):
                value = random.random()
                matrix[i][j] = value
                matrix[j][i] = value

    # Calculate rows for each thread
    rows_per_thread = n // num_threads
    threads = []

    for i in range(num_threads):
        start_row = i * rows_per_thread
        # Ensure the last thread covers any remaining rows
        end_row = (start_row + rows_per_thread) if i < num_threads - 1 else n
        thread = threading.Thread(target=worker, args=(start_row, end_row))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return matrix

def main():
    n = 30000  # Matrix dimension

    print(f"Creating a symmetric random matrix of size {n}x{n} using 2 threads...")

    h = create_symmetric_matrix_parallel(n, num_threads=2)

    print("Hamiltonian matrix created successfully!")

    # Print a small portion of the matrix (optional)
    print("Top-left 5x5 corner of the matrix:")
    for row in h[:5]:
        print(row[:5])

if __name__ == "__main__":
    main()

