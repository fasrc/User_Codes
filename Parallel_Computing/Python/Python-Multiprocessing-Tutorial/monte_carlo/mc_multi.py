# Monte Carlo Simulation Script 2: Multiprocessing Implementation

# File: monte_carlo_multiprocessing.py
import numpy as np
import time
import argparse
from multiprocessing import Pool

def monte_carlo_worker(num_samples):
    count_inside_circle = 0
    for _ in range(num_samples):
        x, y = np.random.rand(), np.random.rand()
        if x ** 2 + y ** 2 <= 1:
            count_inside_circle += 1
    return count_inside_circle

def monte_carlo_pi_multiprocessing(total_samples, num_processes):
    samples_per_process = total_samples // num_processes
    with Pool(num_processes) as pool:
        counts = pool.map(monte_carlo_worker, [samples_per_process] * num_processes)
    return (4 * sum(counts)) / total_samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=1000000, help='Number of samples to generate')
    parser.add_argument('--processes', type=int, default=4, help='Number of processes to use')
    args = parser.parse_args()

    start_time = time.time()
    pi_estimate = monte_carlo_pi_multiprocessing(args.samples, args.processes)
    end_time = time.time()

    print(f"Estimated value of Pi: {pi_estimate}")
    print(f"Execution Time: {end_time - start_time} seconds")

if __name__ == "__main__":
    main()
