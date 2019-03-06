### Purpose:

**Orphaned directives -** This example demonstrates a dot product  being performed by an orphaned loop reduction construct.

### Contents:

* <code>omp_dot2.c</code>: C source code
* <code>omp_dot2.dat</code>: Output file
* <code>Makefile</code>: Makefile to compile the code
* <code>sbatch.run</code>: Batch-job submission script

### Example Usage:

```bash
module load gcc/8.2.0-fasrc01		# Load required software modules
make             			# Compile
sbatch sbatch.run 			# Send the job to the queue
```

### Source Code:

```c
/*
  PROGRAM: omp_orphan.c
  DESCRIPTION:
    OpenMP Example - Parallel region with an orphaned directive
    This example demonstrates a dot product  being performed by
    an orphaned loop reduction construct.
 */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define VECLEN 100

float a[VECLEN], b[VECLEN], sum;

// Function dotprod.........................................
float dotprod (){
  int i,tid, chunk;
  float d1, d2, pdot;
  chunk = 5;
  tid = omp_get_thread_num();
#pragma omp for schedule(static,chunk) reduction(+:sum)
  for ( i = 0; i < VECLEN; i++ ){
    d1 = a[i];
    d2 = b[i];
    sum = sum + ( d1 * d2 );
    pdot = sum;
  }
  printf("Thread %d: partial dot product = %f\n",tid, pdot);
}

// Main program.............................................
int main (int argc, char *argv[]) {
  int i, tid, nthreads;

  for ( i = 0; i < VECLEN; i++ ){
    a[i] = 1.0 * i;
    b[i] = 2.0 * i;
  }
  sum = 0.0;

#pragma omp parallel
  {
    tid = omp_get_thread_num();
    nthreads = omp_get_num_threads();
    if ( tid == 0 ){
      printf("Running on %d threads.\n", nthreads);
    }

    dotprod();
  }
  
  printf("Global dot product = %f\n",sum);

  return 0; 
}
```

### Batch-Job Submission Script:

```bash
#!/bin/bash
#SBATCH -J omp_dot2
#SBATCH -o omp_dot2.out
#SBATCH -e omp_dot2.err
#SBATCH -p shared
#SBATCH -t 0-00:30
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=4000

# Set up environment
WORK_DIR=/scratch/${USER}/${SLURM_JOB_ID}
PRO=omp_dot2
### or WORK_DIR=/n/regal/cs205/${USER}/${SLURM_JOB_ID}
mkdir -pv ${WORK_DIR}
cd $WORK_DIR
cp ${SLURM_SUBMIT_DIR}/${PRO}.x .

# Load required software modules
module load gcc/8.2.0-fasrc01

# Run program
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun -c $SLURM_CPUS_PER_TASK ./${PRO}.x | sort > ${PRO}.dat

# Copy back the result and clean up
cp *.dat ${SLURM_SUBMIT_DIR}
rm -rf ${WORK_DIR}
```

### Example Output:

```bash
> cat omp_dot2.dat 
Global dot product = 656700.000000
Running on 4 threads.
Thread 0: partial dot product = 128300.000000
Thread 1: partial dot product = 150550.000000
Thread 2: partial dot product = 175300.000000
Thread 3: partial dot product = 202550.000000
```

