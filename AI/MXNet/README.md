# MXNet

<img src="Images/mxnet-logo.png" alt="MXNet-logo" width="200"/>

## What is MXNet?

[MXNet](https://mxnet.apache.org) is a deep learning framework designed for both efficiency and flexibility. It allows you to mix symbolic and imperative programming to maximize efficiency and productivity. At its core, MXNet contains a dynamic dependency scheduler that automatically parallelizes both symbolic and imperative operations on the fly. A graph optimization layer on top of that makes symbolic execution fast and memory efficient. MXNet is portable and lightweight, scalable to many GPUs and machines.

## Installing MXNet:

These instructions are intended to help you install MXNet on the FASRC cluster.

### GPU Support

For general information on running GPU jobs refer to our [user documentation](https://www.rc.fas.harvard.edu/resources/documentation/gpgpu-computing-on-the-cluster).

The specific example illustrates the installation of MXNet version 1.8 with Python version 3.8, CUDA version 11.1, and CUDNN version 8.0.4. You may also refer to the [installation instructions](https://mxnet.apache.org/versions/1.8.0/get_started?platform=linux&language=python&processor=gpu&environ=pip&) at the official MXNet website.

(1) Start an interactive job requesting GPUs, e.g.,

```bash
salloc -p gpu -t 0-06:00 --mem=8000 --gres=gpu:1 
```

(2) Load required software modules, e.g.,

```bash
module load python/3.8.5-fasrc01
module load cuda/11.7.1-fasrc01
module load cudnn/8.5.0.96_cuda11-fasrc01
```

(3) Create a [conda environment](https://conda.io/projects/conda/en/latest/index.html), e.g.,

```bash
conda create -n mxnet1.9.1_cuda11 python=3.10 pip numpy wheel matplotlib seaborn pandas jupyterlab
```

(4) Activate the new *conda* environment:

```bash
source activate mxnet1.9.1_cuda11
```

(5) Install [NCCL](https://developer.nvidia.com/nccl) via *conda*

```bash
conda install -c conda-forge nccl=2.14.3.1
```

(6) Install MXNet with pip

```bash
pip install mxnet-cu112
```

### Pull a MXNet Singularity container

Alternatively, you can pull an optimized MXNet container from the [NVIDA NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/mxnet), e.g.,

```bash
singularity pull docker://nvcr.io/nvidia/mxnet:22.08-py3
```

The NGC catalog provides access to optimized containers of many popular apps.
This will result in the image <code>mxnet_22.08-py3.sif</code>. The image then can be used with, e.g.,

```bash
salloc -p gpu -t 0-06:00 --mem=8000 --gres=gpu:1 
```

And then on the compute node:

```python
$ singularity exec --nv mxnet_22.08-py3.sif python 
Python 3.8.10 (default, Jun 22 2022, 20:18:18) 
[GCC 9.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import mxnet as mx
>>> print(mx.__version__)
1.9.1
>>> x = mx.nd.ones((3,4), ctx=mx.gpu()) # Create array of ones at GPU 0
[16:42:54] ../src/storage/storage.cc:196: Using Pooled (Naive) StorageManager for GPU
>>> print(x)

[[1. 1. 1. 1.]
 [1. 1. 1. 1.]
 [1. 1. 1. 1.]]
<NDArray 3x4 @gpu(0)>
>>> x.copyto(mx.gpu(1)) # Copy to GPU 1
[16:47:14] ../src/storage/storage.cc:196: Using Pooled (Naive) StorageManager for GPU

[[1. 1. 1. 1.]
 [1. 1. 1. 1.]
 [1. 1. 1. 1.]]
<NDArray 3x4 @gpu(1)>
```

## Running MXNet:

### Run MXNet Interactively

For an **interactive session** to work with the GPUs you can use following:

```bash
salloc -p gpu -t 0-06:00 --mem=8000 --gres=gpu:1 
```

Load required software modules and source your MXNet conda environment.

```bash
[username@holygpu2c0716 ~]$ module load python/3.8.5-fasrc01 cuda/11.7.1-fasrc01 cudnn/8.5.0.96_cuda11-fasrc01  && source activate mxnet1.9.1_cuda11
(mxnet1.9.1_cuda11)
```

Test MXNet interactively:

```bash
(mxnet1.9.1_cuda11) $ python mxnet_test.py
[15:24:16] ../src/operator/nn/./cudnn/./cudnn_algoreg-inl.h:97: Running performance tests to find the best convolution algorithm, this can take a while... (set the environment variable MXNET_CUDNN_AUTOTUNE_DEFAULT to 0 to disable)
training acc at epoch 0: accuracy=0.945417
training acc at epoch 1: accuracy=0.982333
training acc at epoch 2: accuracy=0.986733
training acc at epoch 3: accuracy=0.990133
training acc at epoch 4: accuracy=0.991383
training acc at epoch 5: accuracy=0.993450
training acc at epoch 6: accuracy=0.994317
training acc at epoch 7: accuracy=0.995267
training acc at epoch 8: accuracy=0.995467
training acc at epoch 9: accuracy=0.995883
training acc at epoch 10: accuracy=0.996467
training acc at epoch 11: accuracy=0.996833
training acc at epoch 12: accuracy=0.996500
training acc at epoch 13: accuracy=0.997117
training acc at epoch 14: accuracy=0.997117
training acc at epoch 15: accuracy=0.997533
training acc at epoch 16: accuracy=0.997433
training acc at epoch 17: accuracy=0.997383
training acc at epoch 18: accuracy=0.997833
training acc at epoch 19: accuracy=0.998067
validation acc: accuracy=0.991300
```

<code>mxnet_test.py</code> performs classification of handwritten digits with the MNIST data-set applying a convolutional algorithm:

```python
#!/usr/bin/env python
from __future__ import print_function
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd as ag
import mxnet.ndarray as F

# --- Define a 2D CNN ---
class Net(gluon.Block):
    def __init__(self, **kwargs):
        super(Net, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(20, kernel_size=(5,5))
        self.pool1 = nn.MaxPool2D(pool_size=(2,2), strides = (2,2))
        self.conv2 = nn.Conv2D(50, kernel_size=(5,5))
        self.pool2 = nn.MaxPool2D(pool_size=(2,2), strides = (2,2))
        self.fc1 = nn.Dense(500)
        self.fc2 = nn.Dense(10)

    def forward(self, x):
        x = self.pool1(F.tanh(self.conv1(x)))
        x = self.pool2(F.tanh(self.conv2(x)))
        # 0 means copy over size from corresponding dimension.
        # -1 means infer size from the rest of dimensions.
        x = x.reshape((0, -1))
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return x

# --- Fixing the random seed ---
mx.random.seed(99)

# --- Load the MNIST dataset ---
mnist = mx.test_utils.get_mnist()

# --- Batch size ---
batch_size = 100

# --- Split data into training and validation sets ---
train_data = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
val_data   = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)

# --- Define the net ---
net = Net()

# --- Set the context on GPU if available, otherwise CPU ---
ctx = [mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()]
net.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.001})

# --- Use Accuracy as the evaluation metric ---
metric = mx.metric.Accuracy()
softmax_cross_entropy_loss = gluon.loss.SoftmaxCrossEntropyLoss()

epoch = 20

for i in range(epoch):
    # Reset the train data iterator.
    train_data.reset()
    # Loop over the train data iterator.
    for batch in train_data:
        # Splits train data into multiple slices along batch_axis
        # and copy each slice into a context.
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        # Splits train labels into multiple slices along batch_axis
        # and copy each slice into a context.
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        outputs = []
        # Inside training scope
        with ag.record():
            for x, y in zip(data, label):
                z = net(x)
                # Computes softmax cross entropy loss.
                loss = softmax_cross_entropy_loss(z, y)
                # Backpropogate the error for one iteration.
                loss.backward()
                outputs.append(z)
        # Updates internal evaluation
        metric.update(label, outputs)
        # Make one step of parameter update. Trainer needs to know the
        # batch size of data to normalize the gradient by 1/batch_size.
        trainer.step(batch.data[0].shape[0])
    # Gets the evaluation result.
    name, acc = metric.get()
    # Reset evaluation result to initial state.
    metric.reset()
    print('training acc at epoch %d: %s=%f'%(i, name, acc))

# --- Use Accuracy as the evaluation metric ---
metric = mx.metric.Accuracy()
# Reset the validation data iterator.
val_data.reset()
# Loop over the validation data iterator.
for batch in val_data:
    # Splits validation data into multiple slices along batch_axis
    # and copy each slice into a context.
    data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
    # Splits validation label into multiple slices along batch_axis
    # and copy each slice into a context.
    label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
    outputs = []
    for x in data:
        outputs.append(net(x))
    # Updates internal evaluation
    metric.update(label, outputs)
print('validation acc: %s=%f'%metric.get())
assert metric.get()[1] > 0.98
```

### Batch Jobs

An example batch-job submission script is included below:

```bash
#!/bin/bash
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -t 0-00:30
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH -o mxnet_%j.out 
#SBATCH -e mxnet_%j.err 

# Load software modules and source conda environment
module load python/3.8.5-fasrc01
module load cuda/11.7.1-fasrc01
module load cudnn/8.5.0.96_cuda11-fasrc01
source activate mxnet1.9.1_cuda11

# Run program
srun -c 1 --gres=gpu:1 python mxnet_test.py 
```

If you name the above batch-job submission script <code>run.sbatch</code>, for instance, the job is submitted with:

```bash
sbatch run.sbatch
```

## References:

* [Official MXNet website](https://mxnet.apache.org)
* [MXNet Tutorials](https://mxnet.apache.org/versions/master/api)
* [Dive into Deep Learning book](https://d2l.ai/index.html)
