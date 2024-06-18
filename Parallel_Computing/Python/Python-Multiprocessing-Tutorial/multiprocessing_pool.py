import multiprocessing
import os
import time

# In order to use multiple cores/CPUs, need to request a compute node using --cpus-per-task flag, as shown:
# salloc --partition=test --nodes=1 --cpus-per-task=10 --mem=10GB --time=00:30:00

def cube(x):
    return x**3

def cubeprocess(x, output):
    cubecalc = x**3
    output.put(cubecalc)

if __name__ == "__main__":

    print(f"Number of CPUs allocated:{len(os.sched_getaffinity(0))}")
    
    # The Process class
    #
    # Define an output queue
    output = multiprocessing.Queue()
    start_time_process = time.perf_counter()

    # Setup a list of processes that we want to run
    #processes = [multiprocessing.Process(target=cube, args=(x,)) for x in range(1,len(os.sched_getaffinity(0)))]
    processes = [multiprocessing.Process(target=cubeprocess, args=(x, output)) for x in range(1,len(os.sched_getaffinity(0)))]

    # Run processes
    [p.start() for p in processes]
    
    # Exit the completed processes
    #result_process = [p.join() for p in processes]
    [p.join() for p in processes]

    # Get process results from the output queue
    result_process = [output.get() for p in processes]
    
    finish_time_process = time.perf_counter()
    print(f"Program with process class finished in {finish_time_process-start_time_process} seconds")
    print(result_process)

    # The Pool class
    #
    start_time_pool = time.perf_counter()
    pool = multiprocessing.Pool(processes=len(os.sched_getaffinity(0)))
    results_pool = pool.map(cube, range(1,len(os.sched_getaffinity(0))))
    finish_time_pool = time.perf_counter()
    print(f"Program with pool class finished in {finish_time_pool-start_time_pool} seconds")
    print(results_pool)



