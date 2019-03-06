### Purpose:

**Parallel Sections in OpenMP -** In this example, the OpenMP SECTION directive is used to assign different array operations to each thread that executes a SECTION.

### Contents:

* <code>omp_sections.c</code>: C source code
* <code>omp_sections.dat</code>: Output file
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
  PROGRAM: omp_sections.c
  DESCRIPTION:
    OpenMP Example - Sections Work-sharing - C Version
    In this example, the OpenMP SECTION directive is used to assign
    different array operations to each thread that executes a SECTION.
 */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define N  50

int main (int argc, char *argv[]) 
{
int i, nthreads, tid;
float a[N], b[N], c[N], d[N];

/* Some initializations */
for (i=0; i<N; i++) {
  a[i] = i * 1.5;
  b[i] = i + 22.35;
  c[i] = d[i] = 0.0;
  }

#pragma omp parallel shared(a,b,c,d,nthreads) private(i,tid)
  {
  tid = omp_get_thread_num();
  if (tid == 0)
    {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }
  printf("Thread %d starting...\n",tid);

  #pragma omp sections nowait
    {
    #pragma omp section
      {
      printf("Thread %d doing section 1\n",tid);
      for (i=0; i<N; i++)
        {
        c[i] = a[i] + b[i];
        printf("Thread %d: c[%d]= %f\n",tid,i,c[i]);
        }
      }

    #pragma omp section
      {
      printf("Thread %d doing section 2\n",tid);
      for (i=0; i<N; i++)
        {
        d[i] = a[i] * b[i];
        printf("Thread %d: d[%d]= %f\n",tid,i,d[i]);
        }
      }

    }  /* end of sections */

    printf("Thread %d done.\n",tid); 

  }  /* end of parallel section */

  return 0;
}
```

### Batch-Job Submission Script:

```bash
#!/bin/bash
#SBATCH -J omp_sections
#SBATCH -o omp_sections.out
#SBATCH -e omp_sections.err
#SBATCH -p shared
#SBATCH -t 0-00:30
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=4000

# Set up environment
WORK_DIR=/scratch/${USER}/${SLURM_JOB_ID}
PRO=omp_sections
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
> cat omp_sections.dat 
Number of threads = 4
Thread 0 done.
Thread 0 starting...
Thread 1: c[0]= 22.350000
Thread 1: c[10]= 47.349998
Thread 1: c[11]= 49.849998
Thread 1: c[1]= 24.850000
Thread 1: c[12]= 52.349998
Thread 1: c[13]= 54.849998
Thread 1: c[14]= 57.349998
Thread 1: c[15]= 59.849998
Thread 1: c[16]= 62.349998
Thread 1: c[17]= 64.849998
Thread 1: c[18]= 67.349998
Thread 1: c[19]= 69.849998
Thread 1: c[20]= 72.349998
Thread 1: c[21]= 74.849998
Thread 1: c[2]= 27.350000
Thread 1: c[22]= 77.349998
Thread 1: c[23]= 79.849998
Thread 1: c[24]= 82.349998
Thread 1: c[25]= 84.849998
Thread 1: c[26]= 87.349998
Thread 1: c[27]= 89.849998
Thread 1: c[28]= 92.349998
Thread 1: c[29]= 94.849998
Thread 1: c[30]= 97.349998
Thread 1: c[31]= 99.849998
Thread 1: c[32]= 102.349998
Thread 1: c[3]= 29.850000
Thread 1: c[33]= 104.849998
Thread 1: c[34]= 107.349998
Thread 1: c[35]= 109.849998
Thread 1: c[36]= 112.349998
Thread 1: c[37]= 114.849998
Thread 1: c[38]= 117.349998
Thread 1: c[39]= 119.849998
Thread 1: c[40]= 122.349998
Thread 1: c[41]= 124.849998
Thread 1: c[42]= 127.349998
Thread 1: c[43]= 129.850006
Thread 1: c[4]= 32.349998
Thread 1: c[44]= 132.350006
Thread 1: c[45]= 134.850006
Thread 1: c[46]= 137.350006
Thread 1: c[47]= 139.850006
Thread 1: c[48]= 142.350006
Thread 1: c[49]= 144.850006
Thread 1: c[5]= 34.849998
Thread 1: c[6]= 37.349998
Thread 1: c[7]= 39.849998
Thread 1: c[8]= 42.349998
Thread 1: c[9]= 44.849998
Thread 1 doing section 1
Thread 1 done.
Thread 1 starting...
Thread 2: d[0]= 0.000000
Thread 2: d[10]= 485.249969
Thread 2: d[11]= 550.274963
Thread 2: d[12]= 618.299988
Thread 2: d[1]= 35.025002
Thread 2: d[13]= 689.324951
Thread 2: d[14]= 763.349976
Thread 2: d[15]= 840.374939
Thread 2: d[16]= 920.399963
Thread 2: d[17]= 1003.424988
Thread 2: d[18]= 1089.449951
Thread 2: d[19]= 1178.474976
Thread 2: d[20]= 1270.500000
Thread 2: d[21]= 1365.524902
Thread 2: d[22]= 1463.549927
Thread 2: d[23]= 1564.574951
Thread 2: d[24]= 1668.599976
Thread 2: d[25]= 1775.625000
Thread 2: d[26]= 1885.649902
Thread 2: d[27]= 1998.674927
Thread 2: d[2]= 73.050003
Thread 2: d[28]= 2114.699951
Thread 2: d[29]= 2233.724854
Thread 2: d[30]= 2355.750000
Thread 2: d[3]= 114.075005
Thread 2: d[31]= 2480.774902
Thread 2: d[32]= 2608.799805
Thread 2: d[33]= 2739.824951
Thread 2: d[34]= 2873.849854
Thread 2: d[35]= 3010.875000
Thread 2: d[36]= 3150.899902
Thread 2: d[37]= 3293.924805
Thread 2: d[38]= 3439.949951
Thread 2: d[39]= 3588.974854
Thread 2: d[40]= 3741.000000
Thread 2: d[41]= 3896.024902
Thread 2: d[4]= 158.100006
Thread 2: d[42]= 4054.049805
Thread 2: d[43]= 4215.074707
Thread 2: d[44]= 4379.100098
Thread 2: d[45]= 4546.125000
Thread 2: d[46]= 4716.149902
Thread 2: d[47]= 4889.174805
Thread 2: d[48]= 5065.199707
Thread 2: d[49]= 5244.225098
Thread 2: d[5]= 205.125000
Thread 2: d[6]= 255.150009
Thread 2: d[7]= 308.175018
Thread 2: d[8]= 364.200012
Thread 2: d[9]= 423.225006
Thread 2 doing section 2
Thread 2 done.
Thread 2 starting...
Thread 3 done.
Thread 3 starting...
```

