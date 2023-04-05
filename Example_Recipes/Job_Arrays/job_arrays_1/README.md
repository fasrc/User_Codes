#### PURPOSE:

Example workflow illustrating use of job arrays and a "master" Python
script to perform a parameter sweep on the FASRC cluster at
Harvard University.

#### CONTENTS:

(1) pro.c: Example C code calculating sum of integers from 1 to N
           where N is an integer parameter read from command line

(2) test.py: "Master" Python script. It does the following:

   * Compiles pro.c program and creates an executable (pro.x)

   * Creates directories dir1, dir2, dir3, dir4, and dir5

   * Copies the executable "pro.x" to these directories

   * Creates a batch job-submission script for an array job (with 5 instances in this example)

   * Saves a copy of the batch-job submission script named "test.run", in current directory

   * Submits an array job to the queue. Each instance of the job array executes "pro.x" 
      with a different value of the parameter N (100, 200, 300, 400, and 500 in this example)
