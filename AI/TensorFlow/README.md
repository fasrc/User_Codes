# TensorFlow
<img src="Images/tensorflow-logo.png" alt="TF-logo" width="200"/>

## What is TensorFlow?

[TensorFlow](https://www.tensorflow.org) (TF) is an open-source software library for numerical computation using data flow graphs. Nodes in the graph represent mathematical operations, while the graph edges represent the multidimensional data arrays (tensors) that flow between them. This flexible architecture lets you deploy computation to one or more CPUs or GPUs in a desktop, server, or mobile device without rewriting code.

TensorFlow was originally developed by researchers and engineers working on the Google Brain team within Google's Machine Intelligence research organization for the purposes of conducting machine learning and deep neural networks research. The system is general enough to be applicable in a wide variety of other domains, as well.

## Installing TensorFlow:

The below instructions are intended to help you set up TF on the FASRC cluster.

### GPU Version

The specific example illustrates the installation of TF version 2.10.0 with Python version 3.8, CUDA version 11.7.1, and CUDNN version 8.5.0.96. Please refer to our documentation on [running GPU jobs on the FASRC cluster](https://www.rc.fas.harvard.edu/resources/documentation/gpgpu-computing-on-the-cluster/).

The two recommended methods for setting up TF in your user environment is installing TF in a [conda environment](https://www.rc.fas.harvard.edu/resources/documentation/software-on-the-cluster/python/) in your user space, or use a TF [singularity container](https://www.rc.fas.harvard.edu/resources/documentation/software/singularity-on-the-cluster).

**NOTE:** If you intend to use TF on [NVIDIA A100 Tensor Core GPU](https://www.nvidia.com/en-us/data-center/a100), please make sure to use TF version 2.5 or greater, with CUDA version 11.1 or greater and CUDNN version 8.1.0 or greater.

**Installing TF in a Conda Environment**

You can install your own TF instance following these simple steps:

* Load required software modules, e.g.,

```bash
module load python/3.8.5-fasrc01
module load cuda/11.7.1-fasrc01
module load cudnn/8.5.0.96_cuda11-fasrc01
```

* Create a new *conda* environment with Python and some additional packages needed by TensorFlow 
(**Note:** the specific example includes additional packages, such as <code>scipy</code>, <code>pandas</code>, <code>matplotlib</code>, <code>seaborn</code>, <code>h5py</code> and <code>jupyterlab</code>, required for data analytics and visualization.)

```bash
$ conda create -n tf2.10_cuda11 python=3.10 pip numpy six wheel scipy pandas matplotlib seaborn h5py jupyterlab
```

* Activate the new *conda* environment, e.g.,

```bash
source activate tf2.10_cuda11
(tf2.10_cuda11) $ pip install tensorflow==2.10
```

* Install TF with <code>pip</code>, e.g.,

```bash
(tf2.10_cuda11) $ pip install --upgrade tensorflow-gpu==2.10
```
**Note:** This will install the TF version 2.10. TensorFlow version 2.12 does not work on the cluster as it needs a newer CUDA driver and cudnn 8.6 so you will need to use TF 2.10. Create a rchelp ticket if you need newer version.  

**Pull a TF singularity container**

Alternatively, one can pull and use a TensorFlow [singularity](https://sylabs.io/guides/3.4/user-guide/index.html) container:

```bash
singularity pull --name tf2.10_gpu.simg docker://tensorflow/tensorflow:latest-gpu
```

This will result in the image <code>tf2.10_gpu.simg</code>. The image then can be used with, e.g.,

```python
$ singularity exec --nv tf2.10_gpu.simg python3
Python 3.8.10 (default, Jun 22 2022, 20:18:18) 
[GCC 9.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
>>> import tensorflow as tf
>>> print(tf.__version__)
2.10.0
>>> print(tf.reduce_sum(tf.random.normal([1000, 1000])))
tf.Tensor(-622.4288, shape=(), dtype=float32)
>>> print(tf.config.list_physical_devices('GPU'))
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:2', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:3', device_type='GPU')]
```

**Note:** Please notice the use of the <code>--nv</code> option. This is required to make use of the NVIDIA GPU card on the host system.

Alternatively, you can pull a container from the [NVIDA NGC Catalog](https://catalog.ngc.nvidia.com), e.g.,

```bash
singularity pull docker://nvcr.io/nvidia/tensorflow:22.08-tf2-py3
```

The NGC catalog provides access to optimized containers of many popular apps.

### CPU Version

Similarly to the GPU installation you can either install TF in a *conda* environment or use a TF singularity container.

**Installing TF in a Conda Environment**

```bash
# (1) Load required software modules
module load python/3.8.5-fasrc01

# (2) Create conda environment
conda create -n tf2.10_cpu python=3.10 pip numpy six wheel scipy pandas matplotlib seaborn h5py jupyterlab

# (3) Activate the conda environment
source activate tf2.10_cpu
(tf2.10_cpu)

# (4) Install TF with pip
pip install --upgrade tensorflow
```
**Note:** This will install the latest TF version. 
If you need a specific version, use, e.g., 
```bash 
pip install --upgrade tensorflow-gpu==2.5
```

**Pull a TF singularity container**

```bash
singularity pull --name tf2.10_cpu.simg docker://tensorflow/tensorflow:latest
```

This will result in the image <code>tf2.10_cpu.simg</code>. The image then can be used with, e.g.,

```python
singularity exec tf2.10_cpu.simg python3 -c "import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'; import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
tf.Tensor(-126.95773, shape=(), dtype=float32)
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
Mon May 24 21:49:01 2021
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 465.19.01    Driver Version: 465.19.01    CUDA Version: 11.3     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA Tesla V1...  On   | 00000000:06:00.0 Off |                    0 |
| N/A   42C    P0    30W / 250W |      0MiB / 32510MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+ 
```

Load required modules, and source your TF environment:

```bash
[username@holygpu2c0701 ~]$ module load python/3.8.5-fasrc01 cuda/11.7.1-fasrc01 cudnn/8.5.0.96_cuda11-fasrc01 && source activate tf2.10_cuda11 
(tf2.10_cuda11) [username@holygpu2c0701 ~]$ 
```

Test TF:

(Example adapted from [here](https://www.tensorflow.org/tutorials/keras/classification/).)

```bash
(tf2.10cuda11) [username@holygpu2c0701 ~]$ python tf_test.py
2.10.0
Epoch 1/10
1875/1875 [==============================] - 4s 2ms/step - loss: 0.4959 - accuracy: 0.8268
Epoch 2/10
1875/1875 [==============================] - 4s 2ms/step - loss: 0.3756 - accuracy: 0.8641
Epoch 3/10
1875/1875 [==============================] - 4s 2ms/step - loss: 0.3351 - accuracy: 0.8760
Epoch 4/10
1875/1875 [==============================] - 4s 2ms/step - loss: 0.3125 - accuracy: 0.8858
Epoch 5/10
1875/1875 [==============================] - 4s 2ms/step - loss: 0.2949 - accuracy: 0.8914
Epoch 6/10
1875/1875 [==============================] - 4s 2ms/step - loss: 0.2803 - accuracy: 0.8968
Epoch 7/10
1875/1875 [==============================] - 4s 2ms/step - loss: 0.2681 - accuracy: 0.9002
Epoch 8/10
1875/1875 [==============================] - 4s 2ms/step - loss: 0.2579 - accuracy: 0.9047
Epoch 9/10
1875/1875 [==============================] - 4s 2ms/step - loss: 0.2491 - accuracy: 0.9074
Epoch 10/10
1875/1875 [==============================] - 4s 2ms/step - loss: 0.2403 - accuracy: 0.9095
313/313 - 1s - loss: 0.3436 - accuracy: 0.8810 - 669ms/epoch - 2ms/step

Test accuracy: 0.8809999823570251
313/313 [==============================] - 0s 1ms/step
[3.74406090e-06 1.09852225e-08 1.39905396e-07 2.27860397e-08
 4.68494136e-07 2.19243602e-03 3.58837468e-07 9.66198891e-02
 1.85329429e-06 9.01181102e-01]
9
(tf2.10_cuda11) [username@holygpu2c0701 ~]$
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


### Examples:

* [Example 1](Example1): Simple 2D CNN with the MNIST dataset
* [Example 2](Example2): TensorBoard application

### References:

* [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
* [TensorFlow API](https://www.tensorflow.org/api_docs/python)
