# TensorFlow
<img src="Images/tensorflow-logo.png" alt="TF-logo" width="200"/>

## Examples:

* [Example 1](Example1): Simple 2D CNN with the MNIST dataset
* [Example 2](Example2): TensorBoard application
* [Example 3](Example3): Multi-gpu example from [TensorFlow documentation](https://www.tensorflow.org/guide/keras/distributed_training)
* [Example 4](Example4): Multi-gpu example -- modified [`tf_test.py`](tf_test.py)

## What is TensorFlow?

[TensorFlow](https://www.tensorflow.org) (TF) is an open-source software library for numerical computation using data flow graphs. Nodes in the graph represent mathematical operations, while the graph edges represent the multidimensional data arrays (tensors) that flow between them. This flexible architecture lets you deploy computation to one or more CPUs or GPUs in a desktop, server, or mobile device without rewriting code.

TensorFlow was originally developed by researchers and engineers working on the Google Brain team within Google's Machine Intelligence research organization for the purposes of conducting machine learning and deep neural networks research. The system is general enough to be applicable in a wide variety of other domains, as well.

## Installing TensorFlow:

The below instructions are intended to help you set up TF on the FASRC cluster.

### GPU Version

The specific example illustrates the installation of TF version 2.16.1 with Python version 3.10, CUDA version 12.1.0, and CUDNN version 9.0.0.312. Please refer to our documentation on [running GPU jobs on the FASRC cluster](https://www.rc.fas.harvard.edu/resources/documentation/gpgpu-computing-on-the-cluster/).

The two recommended methods for setting up TF in your user environment is installing TF in a [conda environment](https://www.rc.fas.harvard.edu/resources/documentation/software-on-the-cluster/python/) in your user space, or use a TF [singularity container](https://www.rc.fas.harvard.edu/resources/documentation/software/singularity-on-the-cluster).

**Installing TF in a Conda Environment**

You can install your own TF instance following these simple steps:

* Load required software modules, e.g.,

```bash
module load python/3.10.13-fasrc01
```

* Create a new `conda` environment with Python: 

```bash
mamba create -n tf2.16.1_cuda12.1 python=3.10 pip wheel
```

* Activate the new `conda` environment, e.g.,

```bash
source activate tf2.16.1_cuda12.1
```

*  Install CUDA and cuDNN with conda/mamba and pip:

```bash
mamba install -c  "nvidia/label/cuda-12.1.0" cuda-toolkit=12.1.0
pip install nvidia-cudnn-cu12==9.0.0.312
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

* Install extra packages required for data analytics, e.g.,

```bash
mamba install -c conda-forge numpy scipy pandas matplotlib seaborn h5py jupyterlab jupyterlab-spellchecker scikit-learn
```

* Install TF plus required GPU libraries with pip, e.g.,

```bash
pip install --upgrade tensorflow[and-cuda]==2.16.*
```  

* Set up the `KERAS` backend (required for `KERAS` version 3.0 and above)

```bash
export KERAS_BACKEND="tensorflow"
```

**NOTE:** Starting with version 2.16.1, TF includes [KERAS version 3.0](https://keras.io/). Please, refer to the [TensorFlow 2.16.1 release notes](https://github.com/tensorflow/tensorflow/releases/tag/v2.16.1) for important changes.

**Pull a TF singularity container**

Alternatively, one can pull and use a TensorFlow [singularity](https://sylabs.io/guides/3.4/user-guide/index.html) container:

```bash
singularity pull --name tf2.16.1_gpu.simg docker://tensorflow/tensorflow:2.16.1-gpu
```

This will result in the image `tf2.16.1_gpu.simg`. The image then can be used with, e.g.,

```python
$ KERAS_BACKEND="tensorflow" singularity exec --nv tf2.16.1_gpu.simg python3
Python 3.11.0rc1 (main, Aug 12 2022, 10:02:14) [GCC 11.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
>>> import tensorflow as tf
>>> print(tf.__version__)
2.16.1
>>> print(tf.reduce_sum(tf.random.normal([1000, 1000])))
tf.Tensor(1365.5554, shape=(), dtype=float32)
>>> print(tf.config.list_physical_devices('GPU'))
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:2', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:3', device_type='GPU')]
```

**Note:** Please notice the use of the <code>--nv</code> option. This is required to make use of the NVIDIA GPU card on the host system. Please also notice the use of `KERAS_BACKEND="tensorflow"` environment variable, which is required to set the KERAS backend to TF.

Alternatively, you can pull a container from the [NVIDA NGC Catalog](https://catalog.ngc.nvidia.com), e.g.,

```bash
singularity pull docker://nvcr.io/nvidia/tensorflow:24.03-tf2-py3
```

This will result in the image `tensorflow_24.03-tf2-py3.sif`, which has TF version `2.15.0`.

The NGC catalog provides access to optimized containers of many popular apps.

### CPU Version

Similarly to the GPU installation you can either install TF in a `conda` environment or use a TF singularity container.

**Installing TF in a Conda Environment**

```bash
# (1) Load required software modules
module load python/3.10.13-fasrc01

# (2) Create conda environment
mamba create -n tf2.16.1_cpu python=3.10 pip wheel

# (3) Activate the conda environment
source activate tf2.16.1_cpu

# (4) Install required packages for data analytics, e.g.,
mamba install -c conda-forge numpy scipy pandas matplotlib seaborn h5py jupyterlab jupyterlab-spellchecker scikit-learn

# (5) Install a CPU version TF with pip
pip install --upgrade tensorflow-cpu==2.16.*

# (6) Set up KERAS backend to use TF
export KERAS_BACKEND="tensorflow"
```

**Pull a TF singularity container**

```bash
singularity pull --name tf2.12_cpu.simg docker://tensorflow/tensorflow:2.12.0
```

This will result in the image <code>tf2.12_cpu.simg</code>. The image then can be used with, e.g.,

```python
KERAS_BACKEND="tensorflow" singularity exec tf2.16.1_cpu.simg python3 -c "import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'; import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
tf.Tensor(2878.413, shape=(), dtype=float32)
```

## Running TensorFlow:

### Run TensorFlow interactively

For an **interactive session** to work with the GPUs you can use following:

```bash
salloc -p gpu_test -t 0-06:00 --mem=8000 --gres=gpu:1
```

While on GPU node, you can run <code>nvidia-smi</code> to get information about the assigned GPU's.

```
[username@holygpu7c26306 ~]$ nvidia-smi
Fri Apr  5 16:00:55 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.12             Driver Version: 535.104.12   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:E3:00.0 Off |                   On |
| N/A   25C    P0              46W / 400W |    259MiB / 40960MiB |     N/A      Default |
|                                         |                      |              Enabled |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| MIG devices:                                                                          |
+------------------+--------------------------------+-----------+-----------------------+
| GPU  GI  CI  MIG |                   Memory-Usage |        Vol|      Shared           |
|      ID  ID  Dev |                     BAR1-Usage | SM     Unc| CE ENC DEC OFA JPG    |
|                  |                                |        ECC|                       |
|==================+================================+===========+=======================|
|  0    2   0   0  |              37MiB / 19968MiB  | 42      0 |  3   0    2    0    0 |
|                  |               0MiB / 32767MiB  |           |                       |
+------------------+--------------------------------+-----------+-----------------------+
                                                                                         
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
[username@holygpu7c26306 ~]$ module load python/3.10.13-fasrc01  && source activate tf2.16.1_cuda12.1 
(tf2.16.1_cuda12.1) [username@holygpu7c26306 ~]$ 
```

Test TF:

(Example adapted from [here](https://www.tensorflow.org/tutorials/keras/classification/).)

```bash
(tf2.16.1_cuda12.1) [username@holygpu7c26306 ~]$ python tf_test.py
2.16.1
Epoch 1/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 839us/step - accuracy: 0.7867 - loss: 0.6247  
Epoch 2/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 829us/step - accuracy: 0.8600 - loss: 0.3855
Epoch 3/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 3s 827us/step - accuracy: 0.8788 - loss: 0.3373   
Epoch 4/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 831us/step - accuracy: 0.8852 - loss: 0.3124
Epoch 5/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 828us/step - accuracy: 0.8912 - loss: 0.2915
Epoch 6/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 3s 830us/step - accuracy: 0.8961 - loss: 0.2773   
Epoch 7/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 828us/step - accuracy: 0.9025 - loss: 0.2625
Epoch 8/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 830us/step - accuracy: 0.9044 - loss: 0.2606
Epoch 9/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 828us/step - accuracy: 0.9081 - loss: 0.2489
Epoch 10/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 829us/step - accuracy: 0.9109 - loss: 0.2405
313/313 - 2s - 6ms/step - accuracy: 0.8804 - loss: 0.3411

Test accuracy: 0.8804000020027161
313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step   
[1.0222636e-07 7.9844620e-09 4.7857565e-11 5.2755653e-09 2.7131367e-10
 2.1757800e-04 5.9717085e-09 6.6847289e-03 4.5007189e-07 9.9309713e-01]
9
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

## References:

* [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
* [TensorFlow API](https://www.tensorflow.org/api_docs/python)
