In this folder, we have a folder named 'directories'. 'directories'
contains multiple sub-folders which contain a job.sh file.  This
example script simply prints 'Hello World' to the console.

In order to process these files, we can use main.sh script as follows:
./main.sh directories f job.sh output

This will generate some .o (output) and .e (error) files along with
output folders starting with the name 'output' and the corresponding
Slurm job and task array IDs appended. 

If you 'cat' or 'more' the contents of .o files, you should see 'Hello
World' in those files along with print statements from the 'echo'
command.

Additionally, the output folders will have output.log that stores the
path of the entry being processed in the joblist.txt file.
