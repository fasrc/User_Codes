# Prime Number Calculation Script 1: Sequential Execution

# File: prime_sequential.py
import time
import argparse
import sys

def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def find_primes_in_range(start, end):
    primes = []
    total_numbers = end - start
    progress_bar = ["."] * 20
    progress_displayed = 0
    for i, num in enumerate(range(start, end)):
        if is_prime(num):
            primes.append(num)
        # Progress output
        current_progress = (i + 1) / total_numbers * 100
        new_progress_displayed = int(current_progress // 5)
        if new_progress_displayed > progress_displayed:
            progress_displayed = new_progress_displayed
            sys.stdout.write(f"\rProgress: [{' '.join(progress_bar[:progress_displayed])}{' ' * (20 - progress_displayed)}]")
            sys.stdout.flush()
    print()  # Move to the next line after progress completion
    return primes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=2, help='Starting number of the range')
    parser.add_argument('--end', type=int, default=100000, help='Ending number of the range')
    args = parser.parse_args()

    start_time = time.time()
    primes = find_primes_in_range(args.start, args.end)
    end_time = time.time()

    print(f"Number of primes found: {len(primes)}")
    print(f"Execution Time: {end_time - start_time} seconds")

if __name__ == "__main__":
    main()

