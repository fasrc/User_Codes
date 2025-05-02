# Monte Carlo Simulation Script 1: Sequential Execution with an Option for Parallel

import numpy as np
import time
import argparse

# Monte Carlo simulation function
def monte_carlo_pi(num_samples):
    count_inside_circle = 0
    for _ in range(num_samples):
        x, y = np.random.rand(), np.random.rand()
        if x ** 2 + y ** 2 <= 1:
            count_inside_circle += 1
    return (4 * count_inside_circle) / num_samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=1000000, help='Number of samples to generate')
    parser.add_argument('--parallel', action='store_true', help='Option to run parallel (using NumPy vectorization)')
    args = parser.parse_args()

    start_time = time.time()
    if args.parallel:
        # Using NumPy vectorized operations to perform Monte Carlo
        num_samples = args.samples
        x, y = np.random.rand(num_samples), np.random.rand(num_samples)
        count_inside_circle = np.sum(x**2 + y**2 <= 1)
        pi_estimate = (4 * count_inside_circle) / num_samples
    else:
        pi_estimate = monte_carlo_pi(args.samples)
    end_time = time.time()

    print(f"Estimated value of Pi: {pi_estimate}")
    print(f"Execution Time: {end_time - start_time} seconds")

if __name__ == "__main__":
    main()
