# TensorFlow
<img src="Images/tensorflow-logo.png" alt="TF-logo" width="200"/>

## What is TensorFlow?

[TensorFlow](https://www.tensorflow.org) (TF) is an open-source software library for numerical computation using data flow graphs. Nodes in the graph represent mathematical operations, while the graph edges represent the multidimensional data arrays (tensors) that flow between them. This flexible architecture lets you deploy computation to one or more CPUs or GPUs in a desktop, server, or mobile device without rewriting code.

TensorFlow was originally developed by researchers and engineers working on the Google Brain team within Google's Machine Intelligence research organization for the purposes of conducting machine learning and deep neural networks research. The system is general enough to be applicable in a wide variety of other domains, as well.

## Installing TensorFlow:

The below instructions are intended to help you set up TF on the FASRC cluster.

### GPU Version

The specific example illustrates the installation of TF version 2.12.0 with Python version 3.10, CUDA version 11.8.0, and CUDNN version 8.6.0.163. Please refer to our documentation on [running GPU jobs on the FASRC cluster](https://www.rc.fas.harvard.edu/resources/documentation/gpgpu-computing-on-the-cluster/).

The two recommended methods for setting up TF in your user environment is installing TF in a [conda environment](https://www.rc.fas.harvard.edu/resources/documentation/software-on-the-cluster/python/) in your user space, or use a TF [singularity container](https://www.rc.fas.harvard.edu/resources/documentation/software/singularity-on-the-cluster).

**Installing TF in a Conda Environment**

You can install your own TF instance following these simple steps:

* Load required software modules, e.g.,

```bash
module load python/3.10.9-fasrc01
```

* Create a new *conda* environment with Python and some additional packages needed by TensorFlow 
(**Note:** the specific example includes additional packages, such as <code>scipy</code>, <code>pandas</code>, <code>matplotlib</code>, <code>seaborn</code>, <code>h5py</code> and <code>jupyterlab</code>, required for data analytics and visualization.)

```bash
mamba create -n tf2.12_cuda11 python=3.10 pip numpy six wheel scipy pandas matplotlib seaborn h5py jupyter jupyterlab
```

* Activate the new *conda* environment, e.g.,

```bash
source activate tf2.12_cuda11
(tf2.12_cuda11) $ 
```

*  Install CUDA and cuDNN with conda/mamba and pip

```bash
(tf2.12_cuda11) $ mamba install -c conda-forge cudatoolkit=11.8.0
(tf2.12_cuda11) $ pip install nvidia-cudnn-cu11==8.6.0.163
```

Configure the system paths. You can do it with the following command every time you start a new terminal after activating your `conda` environment:

```bash
CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib
```

For your convenience it is recommended that you automate it with the following commands. The system paths will be automatically configured when you activate this conda environment:

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

* Install TF with pip, e.g.,

```bash
(tf2.12_cuda11) $ pip install --upgrade tensorflow==2.12.*
```  

> **Note:** Starting with [TF version 2.12](https://github.com/tensorflow/tensorflow/releases/tag/v2.12.0), the redundant packages `tensorflow-gpu` and `tf-nightly-gpu` have been removed. These packages were removed and replaced with packages that direct users to switch to `tensorflow` or `tf-nightly` respectively. 

> **Important:** In Rocky 8, you may encounter the following error:

```bash
Can't find libdevice directory ${CUDA_DIR}/nvvm/libdevice.
...
Couldn't invoke ptxas --version
...
InternalError: libdevice not found at ./libdevice.10.bc [Op:__some_op]
```

To fix this error, you will need to run the following commands once:

```bash
# Install NVCC
mamba install -c nvidia cuda-nvcc=11.3.58
# Configure the XLA cuda directory
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
printf 'export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/lib/\n' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
# Copy libdevice file to the required path
mkdir -p $CONDA_PREFIX/lib/nvvm/libdevice
cp $CONDA_PREFIX/lib/libdevice.10.bc $CONDA_PREFIX/lib/nvvm/libdevice/
```

**Pull a TF singularity container**

Alternatively, one can pull and use a TensorFlow [singularity](https://sylabs.io/guides/3.4/user-guide/index.html) container:

```bash
singularity pull --name tf2.12_gpu.simg docker://tensorflow/tensorflow:2.12.0-gpu
```

This will result in the image `tf2.12_gpu.simg`. The image then can be used with, e.g.,

```python
$ singularity exec --nv tf2.12_gpu.simg python3
Python 3.8.10 (default, Mar 13 2023, 10:26:41) 
[GCC 9.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
>>> import tensorflow as tf
>>> print(tf.__version__)
2.12.0
>>> print(tf.reduce_sum(tf.random.normal([1000, 1000])))
tf.Tensor(2061.4414, shape=(), dtype=float32)
>>> print(tf.config.list_physical_devices('GPU'))
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:2', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:3', device_type='GPU')]
```

**Note:** Please notice the use of the <code>--nv</code> option. This is required to make use of the NVIDIA GPU card on the host system.

Alternatively, you can pull a container from the [NVIDA NGC Catalog](https://catalog.ngc.nvidia.com), e.g.,

```bash
singularity pull docker://nvcr.io/nvidia/tensorflow:23.02-tf2-py3
```

This will result in the image `tensorflow_23.02-tf2-py3.sif`, which has TF version `2.11.0`.

The NGC catalog provides access to optimized containers of many popular apps.

### CPU Version

Similarly to the GPU installation you can either install TF in a *conda* environment or use a TF singularity container.

**Installing TF in a Conda Environment**

```bash
# (1) Load required software modules
module load python/3.10.9-fasrc01

# (2) Create conda environment
mamba create -n tf2.12_cpu python=3.10 pip numpy six wheel scipy pandas matplotlib seaborn h5py jupyterlab

# (3) Activate the conda environment
source activate tf2.12_cpu
(tf2.12_cpu)

# (4) Install TF with pip
pip install --upgrade tensorflow==2.12.*
```

**Pull a TF singularity container**

```bash
singularity pull --name tf2.12_cpu.simg docker://tensorflow/tensorflow:2.12.0
```

This will result in the image <code>tf2.12_cpu.simg</code>. The image then can be used with, e.g.,

```python
singularity exec tf2.12_cpu.simg python3 -c "import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'; import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
tf.Tensor(-727.81104, shape=(), dtype=float32)
```

## Running TensorFlow:

### Run TensorFlow interactively

For an **interactive session** to work with the GPUs you can use following:

```bash
salloc -p gpu_test -t 0-06:00 --mem=8000 --gres=gpu:1
```

While on GPU node, you can run <code>nvidia-smi</code> to get information about the assigned GPU's.

```
[username@holygpu2c0701 ~]$ nvidia-smi
$ nvidia-smi
Wed Jun 21 11:07:21 2023       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 530.30.02              Driver Version: 530.30.02    CUDA Version: 12.1     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                  Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf            Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  Tesla V100-PCIE-32GB            On | 00000000:D8:00.0 Off |                    0 |
| N/A   35C    P0               26W / 250W|      0MiB / 32768MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
```

Load required modules, and source your TF environment:

```bash
[username@holygpu2c0701 ~]$ module load python/3.10.9-fasrc01  && source activate tf2.12_cuda11 
(tf2.12_cuda11) [username@holygpu2c0701 ~]$ 
```

Test TF:

(Example adapted from [here](https://www.tensorflow.org/tutorials/keras/classification/).)

```bash
(tf2.12cuda11) [username@holygpu2c0701 ~]$ python tf_test.py
2.12.0
Epoch 1/10
1875/1875 [==============================] - 11s 5ms/step - loss: 0.5003 - accuracy: 0.8250
Epoch 2/10
1875/1875 [==============================] - 8s 5ms/step - loss: 0.3771 - accuracy: 0.8648
Epoch 3/10
1875/1875 [==============================] - 9s 5ms/step - loss: 0.3399 - accuracy: 0.8773
Epoch 4/10
1875/1875 [==============================] - 9s 5ms/step - loss: 0.3159 - accuracy: 0.8849
Epoch 5/10
1875/1875 [==============================] - 9s 5ms/step - loss: 0.2945 - accuracy: 0.8919
Epoch 6/10
1875/1875 [==============================] - 9s 5ms/step - loss: 0.2820 - accuracy: 0.8963
Epoch 7/10
1875/1875 [==============================] - 9s 5ms/step - loss: 0.2697 - accuracy: 0.8999
Epoch 8/10
1875/1875 [==============================] - 9s 5ms/step - loss: 0.2586 - accuracy: 0.9029
Epoch 9/10
1875/1875 [==============================] - 9s 5ms/step - loss: 0.2488 - accuracy: 0.9077
Epoch 10/10
1875/1875 [==============================] - 9s 5ms/step - loss: 0.2400 - accuracy: 0.9109
313/313 - 1s - loss: 0.3402 - accuracy: 0.8834 - 1s/epoch - 4ms/step

Test accuracy: 0.883400022983551
313/313 [==============================] - 1s 2ms/step
[1.6947217e-06 6.1797940e-09 2.0763697e-09 1.8757864e-09 1.4044283e-08
 3.3252352e-04 1.2653798e-05 1.1238558e-02 3.1918006e-07 9.8841417e-01]
9
(tf2.12_cuda11) [username@holygpu2c0701 ~]$
```

In the above example we used the following test code, <code>tf_test.py</code>:

```python
#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
import numpy as np

print(tf.__version__)
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0
model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28, 28)),
	keras.layers.Dense(128, activation='relu'),
	keras.layers.Dense(10, activation='softmax')
   ])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

predictions = model.predict(test_images)
print(predictions[0])
print(np.argmax(predictions[0]))
```

## TensorFlow Singularity Image from Definition File

You may pull a singularity TensorFlow version 2.12.0 image with the below command:

```bash
# Pull a singularity container with version 2.12.0
singularity pull --name tf2.12_gpu.simg docker://tensorflow/tensorflow:2.12.0-gpu
```
This image comes with a number of basic Python packages. If you need additional packages, you could use the example singularity definition file `tf-2.12.def` to build the singularity image:

```
Bootstrap: docker
From: tensorflow/tensorflow:2.12.0-gpu

%post
    pip install --upgrade pip
    pip install matplotlib
    pip install seaborn
    pip install scipy
    pip install scikit-learn
    pip install jupyterlab
    pip install notebook
```

You could install additional packages directly in the image with `pip` by adding them in the `%post` section of the definition file as illustrated above. Please, refer to our documentation on how to [build singularity images from definition files](https://docs.rc.fas.harvard.edu/kb/singularity-on-the-cluster/#articleTOC_15).

## Examples:

* [Example 1](Example1): Simple 2D CNN with the MNIST dataset
* [Example 2](Example2): TensorBoard application

## References:

* [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
* [TensorFlow API](https://www.tensorflow.org/api_docs/python)
