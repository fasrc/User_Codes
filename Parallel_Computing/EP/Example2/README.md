# Example 2: Serial Summation Example

A basic example demonstrating how to run serial Python code on an HPC cluster using SLURM workload manager.

## Learning Objectives
- Writing SLURM batch scripts
- Running serial Python programs on HPC
- Using environment variables for program inputs
- Managing job outputs

## Prerequisites
- Access to SLURM-based HPC cluster
- Python 3.10+
- Basic Python programming knowledge
- Understanding of shell scripting

## Files
- `run.sbatch`: SLURM batch script that runs the program with different inputs
- `serial_sum.py`: Python program that calculates sum of integers from 1 to N

## Key Concepts Illustrated
- Job scheduling with SLURM
- Resource allocation (cores, memory, time)
- Serial computation
- Parameter sweeping using shell loops
- Job output management

## Directory Structure
```
.
├── run.sbatch      # SLURM batch script
├── serial_sum.py   # Main Python program
└── README.md       # This documentation
```

The program will generate additional output files:
- `test_job.out`: Standard output
- `test_job.err`: Error log
- `output_[N].out`: Results for each input N
