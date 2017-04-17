### Purpose:

Program illustrates text rendering with LaTeX.

### Contents:

(1) tex_demo.py: Python source code

(2) run.sbatch: Batch-job submission script for sending the job to the queue

### Example Usage:

```	
source new modules.sh
module load python/2.7.6-fasrc01
module load dvipng/1.14-fasrc01
sbatch run.sbatch
```
	
### Example Output:

The code generates a PNG figure, tex_demo.png

![Output image](tex_demo.png)

### References:

* [Text rendering with LaTeX](http://matplotlib.org/users/usetex.html)

* [Writing mathematical expressions](http://matplotlib.org/users/mathtext.html#mathtext-tutorial)


