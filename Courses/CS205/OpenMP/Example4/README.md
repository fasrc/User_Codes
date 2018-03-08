### Purpose:

**Reduction - parallel dot product -** This example demonstrates a sum reduction within a combined parallel loop construct.

### Contents:

* <code>omp_dot.c</code>: C source code
* <code>omp_dot.dat</code>: Output file
* <code>Makefile</code>: Makefile to compile the code
* <code>sbatch.run</code>: Batch-job submission script

### Example Usage:

```bash
source new-modules.sh				# Set up Harvard's LMOD environment
module load gcc/4.8.2-fasrc01		# Load required software modules
make             					# Compile
sbatch sbatch.run 					# Send the job to the queue
```

### Source Code:

```c
/*
  PROGRAM: omp_dot.c
  DESCRIPTION:
    OpenMP Example - Combined Parallel Loop Reduction
    This example demonstrates a sum reduction within 
    a combined parallel loop construct.
 */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
main(int argc, char *argv[])  {

  int n = 100;
  int i, chunk, tid, nthreads;
  float a[n], b[n], d1, d2, pdot, ddot;
  
  /* Some initializations */
  chunk = 5;
  ddot = 0.0;
  for (i=0; i < n; i++) {
    a[i] = i*1.0;
    b[i] = i*2.0;
  }

#pragma omp parallel default(shared) private(i,d1,d2,tid,pdot)	\
  reduction(+:ddot)  
  {
    tid = omp_get_thread_num();
    nthreads = omp_get_num_threads();
    if ( tid == 0 ){
      printf("Running on %d threads.\n", nthreads);
    }
#pragma omp for schedule(static,chunk)
    for ( i = 0; i < n; i++ ) {
      d1 = a[i];
      d2 = b[i];
      pdot = pdot + ( d1 * d2 );
    }
    printf("Thread %d: partial dot product = %f\n",tid, pdot);
    ddot = ddot + pdot;
  }
  printf("Global dot product = %f\n",ddot);

 return 0;
}
```

### Batch-Job Submission Script:

```bash
#!/bin/bash
#SBATCH -J omp_dot
#SBATCH -o omp_dot.out
#SBATCH -e omp_dot.err
#SBATCH -p shared
#SBATCH -t 0-00:30
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=4000

# Set up environment
WORK_DIR=/scratch/${USER}/${SLURM_JOB_ID}
PRO=omp_dot
### or WORK_DIR=/n/regal/cs205/${USER}/${SLURM_JOB_ID}
mkdir -pv ${WORK_DIR}
cd $WORK_DIR
cp ${SLURM_SUBMIT_DIR}/${PRO}.x .

# Load required software modules
source new-modules.sh
module load gcc/4.8.2-fasrc01

# Run program
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun -c $SLURM_CPUS_PER_TASK ./${PRO}.x | sort > ${PRO}.dat

# Copy back the result and clean up
cp *.dat ${SLURM_SUBMIT_DIR}
rm -rf ${WORK_DIR}
```

### Example Output:

```bash
> cat omp_dot.dat 
Global dot product = 657096.000000
Running on 4 threads.
Thread 0: partial dot product = 128399.000000
Thread 1: partial dot product = 150649.000000
Thread 2: partial dot product = 175399.000000
Thread 3: partial dot product = 202649.000000
```

