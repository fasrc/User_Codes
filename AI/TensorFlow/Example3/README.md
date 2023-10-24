## Purpose

Show how to use multiple GPUs with Tensorflow

## Contents

- `tf_multi_gpu.py`: Simple example from [TensforFlow docs](https://www.tensorflow.org/guide/keras/distributed_training)
- `run.sbatch`: Slurm batch-job submission script to pull singularity image and run `tf_multi_gpu.py`
- `tf_multi.err`: Output file

## Important notes

1. In this example the slurm batch script pulls a singularity container with TensorFlow and runs the examples inside the singularity container. However, you can modify `run.sbatch` script to run within a conda/mamba environment.

2. Even when the code runs successfuly, errors may be printed to the standard error file `tf_multi.err`. For example:

```
2023-10-24 19:23:54.822118: E tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:9342] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2023-10-24 19:23:54.822162: E tensorflow/compiler/xla/stream_executor/cuda/cuda_fft.cc:609] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2023-10-24 19:23:54.826110: E tensorflow/compiler/xla/stream_executor/cuda/cuda_blas.cc:1518] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2023-10-24 19:23:55.224814: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-10-24 19:23:59.486160: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1886] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 31141 MB memory:  -> device: 0, name: Tesla V100-PCIE-32GB, pci bus id: 0000:06:00.0, compute capability: 7.0
2023-10-24 19:23:59.487094: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1886] Created device /job:localhost/replica:0/task:0/device:GPU:1 with 31141 MB memory:  -> device: 1, name: Tesla V100-PCIE-32GB, pci bus id: 0000:2f:00.0, compute capability: 7.0
2023-10-24 19:23:59.487942: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1886] Created device /job:localhost/replica:0/task:0/device:GPU:2 with 31141 MB memory:  -> device: 2, name: Tesla V100-PCIE-32GB, pci bus id: 0000:86:00.0, compute capability: 7.0
2023-10-24 19:23:59.488764: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1886] Created device /job:localhost/replica:0/task:0/device:GPU:3 with 31141 MB memory:  -> device: 3, name: Tesla V100-PCIE-32GB, pci bus id: 0000:d8:00.0, compute capability: 7.0
2023-10-24 19:24:05.807968: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x14f675ac6710 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2023-10-24 19:24:05.808018: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Tesla V100-PCIE-32GB, Compute Capability 7.0
2023-10-24 19:24:05.808031: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (1): Tesla V100-PCIE-32GB, Compute Capability 7.0
2023-10-24 19:24:05.808041: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (2): Tesla V100-PCIE-32GB, Compute Capability 7.0
2023-10-24 19:24:05.808051: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (3): Tesla V100-PCIE-32GB, Compute Capability 7.0
2023-10-24 19:24:05.821196: I tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc:269] disabling MLIR crash reproducer, set env var `MLIR_CRASH_REPRODUCER_DIRECTORY` to enable.
2023-10-24 19:24:05.859640: I tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:442] Loaded cuDNN version 8600
2023-10-24 19:24:05.863727: I tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:442] Loaded cuDNN version 8600
2023-10-24 19:24:05.867418: I tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:442] Loaded cuDNN version 8600
2023-10-24 19:24:05.870285: I tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:442] Loaded cuDNN version 8600
2023-10-24 19:24:05.973099: I ./tensorflow/compiler/jit/device_compiler.h:186] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
```

