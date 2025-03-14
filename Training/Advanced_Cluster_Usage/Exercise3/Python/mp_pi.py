#!/usr/bin/env python3
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Program: mp_pi.py
#          Parallel Monte-Carlo algorithm for calculating PI using multiprocessing
#          Translated from omp_pi.c
#
# Usage:   python mp_pi.py <number_of_samples> <number_of_processes>
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import sys
import math
import time
from multiprocessing import Pool

def lcg_rand(seed):
    a = 69069
    c = 1
    m = 2147483647
    seed = (a * seed + c) % m
    return seed, float(seed) / m

def calc_pi_chunk(args):
    samples, seed_base, process_id = args
    count = 0
    seed = seed_base + process_id * 1999
    
    for i in range(samples):
        seed, rx = lcg_rand(seed)
        seed, ry = lcg_rand(seed)
        x = rx
        y = ry
        z = x * x + y * y
        
        if z <= 1.0:
            count += 1
    
    return count

def main():
    if len(sys.argv) != 3:
        print("Usage: python mp_pi.py <number_of_samples> <number_of_processes>")
        sys.exit(1)
    
    total_samples = int(sys.argv[1])
    nprocesses = int(sys.argv[2])
    
    samples_per_process = total_samples // nprocesses
    remainder = total_samples % nprocesses
    tasks = []
    base_seed = 1202107158
    
    for i in range(nprocesses):
        chunk_size = samples_per_process + (1 if i < remainder else 0)
        tasks.append((chunk_size, base_seed, i))
    
    t0 = time.perf_counter()
    with Pool(processes=nprocesses) as pool:
        results = pool.map(calc_pi_chunk, tasks)
    
    total_count = sum(results)
    tf = time.perf_counter() - t0
    
    pi_estimate = 4.0 * total_count / total_samples
    
    print(f"Number of processes: {nprocesses:2d}")
    print(f"Exact value of PI: {math.pi:7.5f}")
    print(f"Estimate of PI:    {pi_estimate:7.5f}")
    print(f"Time: {tf:7.2f} sec.")

if __name__ == "__main__":
    main()
