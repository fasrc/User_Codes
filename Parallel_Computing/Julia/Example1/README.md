This example demonstrates basic usage of MPI in Julia.

### Contents
- <code>hello_world_mpi.jl</code>: Julia source code
- <code>hello_world_mpi.sh</code>: slurm submission script.

### Example Usage:

```bash
sbatch hello_world_mpi.sh
```
 
### Example Output:

```bash
$ cat hello_world_mpi.out
Hello world, I am rank 3 of 4
Hello world, I am rank 2 of 4
Hello world, I am rank 0 of 4
Hello world, I am rank 1 of 4
```