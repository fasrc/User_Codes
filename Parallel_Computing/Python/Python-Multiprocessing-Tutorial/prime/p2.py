# Prime Number Calculation Script 2: Multiprocessing Implementation

# File: prime_multiprocessing.py
import time
import argparse
import sys
from multiprocessing import Pool, Value, Lock

progress = Value('i', 0)
progress_lock = Lock()

def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def find_primes_in_subrange(subrange):
    start, end = subrange
    primes = []
    total_numbers = end - start
    for i, num in enumerate(range(start, end)):
        if is_prime(num):
            primes.append(num)
        # Update progress
        with progress_lock:
            progress.value += 1
            current_progress = (progress.value / total_numbers) * 100
            if progress.value % (total_numbers // 20) == 0 or progress.value == total_numbers:
                dots = progress.value // (total_numbers // 20)
                sys.stdout.write(f"\rProgress: [{' '.join(['.'] * dots)}{' ' * (20 - dots)}]")
                sys.stdout.flush()
    return primes

def find_primes_in_range_multiprocessing(start, end, num_processes):
    step = (end - start) // num_processes
    subranges = [(start + i * step, start + (i + 1) * step) for i in range(num_processes)]
    subranges[-1] = (subranges[-1][0], end)  # Adjust the last subrange to include the remainder

    with Pool(num_processes) as pool:
        results = pool.map(find_primes_in_subrange, subranges)
    
    primes = [prime for sublist in results for prime in sublist]
    return primes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=2, help='Starting number of the range')
    parser.add_argument('--end', type=int, default=100000, help='Ending number of the range')
    parser.add_argument('--processes', type=int, default=4, help='Number of processes to use')
    args = parser.parse_args()

    start_time = time.time()
    global progress
    progress.value = 0  # Reset progress before starting
    primes = find_primes_in_range_multiprocessing(args.start, args.end, args.processes)
    end_time = time.time()

    print(f"\nNumber of primes found: {len(primes)}")
    print(f"Execution Time: {end_time - start_time} seconds")

if __name__ == "__main__":
    main()
