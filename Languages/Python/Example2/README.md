### Introduction:

Program illustrates text rendering with LaTeX and generating figures on the cluster. Please, notice you need to include the below lines in your Python script

```python
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
```

### Contents:

(1) tex_demo.py: Python source code

(2) run.sbatch: Batch-job submission script for sending the job to the queue

(3) tex_demo.png: Output figure

### Example Usage:

```	bash
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


