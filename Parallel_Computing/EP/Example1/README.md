# Example 1: Job Arrays in SLURM

Demonstrates parallel execution using SLURM job arrays through a simple summation task.

## Prerequisites
- SLURM-based HPC cluster
- Python 3.10+
- Basic SLURM knowledge

## Files
- `serial_sum.py`: Computes sum from 1 to N
- `run.sbatch`: SLURM job array configuration
- `output_{N}.out`: Results for N=100,200,300

## Guide

1. Review the computation:
```python
def serial_sum(x):
    k = 0; s = 0
    while k < x:
        k = k + 1
        s = s + k
    return s
```

2. Submit parallel jobs:
```bash
sbatch run.sbatch
```

3. Monitor execution:
```bash
squeue -u $USER
```

4. Verify results in output files:
- 100: 5050
- 200: 20100
- 300: 45150

## Key Concepts
- Task distribution via job arrays
- Environment variables for task identification
- Independent parallel execution
- Resource allocation per task
