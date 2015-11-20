#!/usr/bin/env python
#=====================================================================
# Program: test.py
#=====================================================================
import subprocess
import time
from popen2 import popen2

# Parameters
pro = "pro.c"  # Name of C program
exe = "pro.x"  # Name of executable

# Job parameters
job_name      = "array_test"         # Job name
job_queue     = "serial_requeue"     # Queue name
job_memory    = 4000                 # Memory ( in MB )
wall_time     = 30                   # Time ( in min )
jobs_in_array = 5                    # Number of jobs in the array
cpus          = 1                    # Number of compute cores
nodes         = 1                    # Number of compute nodes

# Compile the C program
command1 = ["gcc", "-o", exe, pro, "-O2"]
subprocess.call(command1)

# Create directories and copy executable to these directories
for i in range ( 1, 6 ):
    istr = str(i)
    dir_name = "dir"+istr
    command2 = ["mkdir", "-p", dir_name]
    command3 = ["cp", exe, dir_name]
    subprocess.call(command2)
    subprocess.call(command3)
    
# Remove executable from main directory
command4 = ["rm", "-r", exe]
subprocess.call(command4)

# Open a pipe to the sbatch command
output, input = popen2('sbatch')

# Build the job_string
line1  = "#!/bin/bash\n"
line2  = "#SBATCH -J "        + job_name             + "\n"
line3  = "#SBATCH -p "        + job_queue            + "\n"
line4  = "#SBATCH -t "        + str(wall_time)       + "\n"
line5  = "#SBATCH -o "        + "dir%a/" + job_name  + "_%a.out\n"
line6  = "#SBATCH -e "        + "dir%a/" + job_name  + "_%a.err\n"
line7  = "#SBATCH -n "        + str(cpus)            + "\n"
line8  = "#SBATCH -N "        + str(nodes)           + "\n"
line9  = "#SBATCH --mem="     + str(job_memory)      + "\n"
line10 = "#SBATCH --array=1-" + str(jobs_in_array)   + "\n"
line11 = "par=$((${SLURM_ARRAY_TASK_ID}*100))\n"
line12 = "dir${SLURM_ARRAY_TASK_ID}/" + exe +  " $par" + "\n"

job_string = line1 + line2 + line3 + line4   \
           + line5 + line6 + line7 + line8   \
           + line9 + line10 + line11 + line12

# Send job_string to sbatch
input.write(job_string)
input.close()

# Print your job and the system response to the screen as it's submitted
print job_string
print output.read()

# Print out a copy of the batch-job submission script to an external file
f = open("test.run", "w")
f.write(job_string)
f.close()

# Exit
quit()
