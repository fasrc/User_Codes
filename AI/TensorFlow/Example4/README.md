## Purpose

Show how to use multiple GPUs with Tensorflow

## Contents

- `tf_test_multi_gpu.py`: Modified code [`tf_test.py`](../tf_test.py) to use all available GPUs on a node
- `run.sbatch`: Slurm batch-job submission script to pull singularity image and run `tf_test_multi_gpu.py`
- `tf_test.out`: Output file

## Important notes

1. In this example the slurm batch script pulls a singularity container with TensorFlow and runs the examples inside the singularity container. However, you can modify `run.sbatch` script to run within a conda/mamba environment.

