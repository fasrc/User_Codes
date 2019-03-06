### Purpose:

**Parallel region construct -** In this simple example, the master thread forks a parallel region.
All threads in the team obtain their unique thread number and print it.
The master thread only prints the total number of threads.

### Contents:

* <code>omp_hello.c</code>: C source code
* <code>omp_hello.dat</code>: Output file
* <code>Makefile</code>: Makefile to compile the code
* <code>sbatch.run</code>: Batch-job submission script

### Example Usage:

```bash
module gcc/8.2.0-fasrc01	   	        # Load required software modules
make             				# Compile
sbatch sbatch.run 				# Send the job to the queue
```

### Source Code:

```c
/*
  PROGRAM: omp_hello.c
  DESCRIPTION: 
    In this simple example, the master thread forks a parallel region.
    All threads in the team obtain their unique thread number and print it.
    The master thread only prints the total number of threads.  Two OpenMP
    library routines are used to obtain the number of threads and each
    thread's number.
*/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
int main (int argc, char *argv[]) {
  int nthreads;
  int tid;
  // Parallel region starts here............................
#pragma omp parallel private(nthreads,tid)
  {    
    // Get thread ID........................................
    tid = omp_get_thread_num();
    printf("Hello World from thread = %d\n", tid);
    if ( tid == 0 ){
      // Get total number of threads........................
      nthreads = omp_get_num_threads();
      printf("Number of threads = %d\n", nthreads);
    }
  }
  // End of parallel region.................................
}
```

### Batch-Job Submission Script:

```bash
#!/bin/bash
#SBATCH -J omp_hello
#SBATCH -o omp_hello.out
#SBATCH -e omp_hello.err
#SBATCH -p shared
#SBATCH -t 0-00:30
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=4000

# Set up environment
WORK_DIR=/scratch/${USER}/${SLURM_JOB_ID}
PRO=omp_hello
### or WORK_DIR=/n/regal/cs205/${USER}/${SLURM_JOB_ID}
mkdir -pv ${WORK_DIR}
cd $WORK_DIR
cp ${SLURM_SUBMIT_DIR}/${PRO}.x .

# Load required software modules
source new-modules.sh
module load gcc/8.2.0-fasrc01

# Run program
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun -c $SLURM_CPUS_PER_TASK ./${PRO}.x > ${PRO}.dat

# Copy back the result and clean up
cp *.dat ${SLURM_SUBMIT_DIR}
rm -rf ${WORK_DIR}
```

### Example Output:

```bash
> cat omp_hello.dat
Hello World from thread = 0
Number of threads = 4
Hello World from thread = 1
Hello World from thread = 2
Hello World from thread = 3
```

