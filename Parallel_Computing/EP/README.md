# HPC Job Submission Examples

Two examples demonstrating different approaches to parallel task execution in SLURM.

## Example 1: Job Arrays
Uses SLURM job arrays to run multiple tasks in parallel. Each task computes the sum of integers from 1 to N (N=100,200,300).

Key features:
- Parallel execution via `--array`
- Task identification through `SLURM_ARRAY_TASK_ID`
- Independent resource allocation per task

Best for: Parameter sweeps, embarrassingly parallel tasks, multiple independent computations.

## Example 2: Sequential Tasks
Runs multiple tasks sequentially within a single SLURM job using a for-loop and srun.

Key features:
- Sequential execution via bash loop
- Task parameter passing through custom environment variable
- Single job allocation

Best for: Learning SLURM basics, tasks with dependencies, resource sharing between tasks.

## Prerequisites
- SLURM-based HPC cluster
- Python 3.10+
- Basic understanding of HPC concepts

## Directory Structure
```
.
├── README.md
├── example1/
│   ├── run.sbatch       # Job array submission
│   ├── serial_sum.py    # Computation script
│   └── README.md        # Detailed guide
└── example2/
    ├── run.sbatch       # Sequential submission
    ├── serial_sum.py    # Computation script
    └── README.md        # Detailed guide
```

Choose Example 1 for parallel computing concepts or Example 2 for basic SLURM job management.
