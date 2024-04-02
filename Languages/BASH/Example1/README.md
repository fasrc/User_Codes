#### Purpose:

This example illustrates the use of Bash scripts on the Harvard University FAS cluster. The specific example computes the integer sum from 1 to N, where N is a number read from the command line.

#### Contents:

* <code>sum.sh</code>: BASH source code
* <code>run.sbatch</code>: Batch-job submission script
* <code>bash_sum.out</code>: Output file

#### BASH source code:

```bash
#!/bin/bash
#=================================================
# Program: sum.sh
#          Sum up integers from 1 to N, where N
#          is read from the command line
#
# Example usage: sh sum.sh 100
#=================================================
n=$1
k=0
for i in `seq 1 $n`
do
    k=$(($k+$i))
done
echo -e "Sum of integers from 1 to $n is $k."
```

#### Example Batch-Job Submission Script:

```bash
#!/bin/bash
#SBATCH -J bash_sum
#SBATCH -o bash_sum.out
#SBATCH -e bash_sum.err
#SBATCH -p shared
#SBATCH -c 1
#SBATCH -t 0-00:30
#SBATCH --mem=2G

# Run the program
sh sum.sh 100
```

#### Example Usage:

```bash
sbatch run.sbatch
```

#### Example Output:

```bash
$ cat bash_sum.out 
Sum of integers from 1 to 100 is 5050.
```


