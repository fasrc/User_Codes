## TensorFlow
<img src="Images/tensorflow-logo.png" alt="TF-logo" width="200"/>

### What is TensorFlow?

[TensorFlow](https://www.tensorflow.org) (TF) is an open-source software library for numerical computation using data flow graphs. Nodes in the graph represent mathematical operations, while the graph edges represent the multidimensional data arrays (tensors) that flow between them. This flexible architecture lets you deploy computation to one or more CPUs or GPUs in a desktop, server, or mobile device without rewriting code.

TensorFlow was originally developed by researchers and engineers working on the Google Brain team within Google's Machine Intelligence research organization for the purposes of conducting machine learning and deep neural networks research. The system is general enough to be applicable in a wide variety of other domains, as well.

### Installing TensorFlow:

The below instructions are intended to help you set up TF on the FASRC cluster.

#### GPU Version

The specific example illustrates the installation of TF version 2.5.0 with Python version 3.8, CUDA version 11.1, and CUDNN version 8.1.0. Please refer to our documentation on [running GPU jobs on the FASRC cluster](https://www.rc.fas.harvard.edu/resources/documentation/gpgpu-computing-on-the-cluster/).

The two recommended methods for setting up TF in your user environment is installing TF in a [conda environment](https://www.rc.fas.harvard.edu/resources/documentation/software-on-the-cluster/python/) in your user space, or use a TF [singularity container](https://www.rc.fas.harvard.edu/resources/documentation/software/singularity-on-the-cluster).

**NOTE:** *If you intend to use TF on [NVIDIA A100 Tensor Core GPU](https://www.nvidia.com/en-us/data-center/a100), please make sure to use the latest TF version 2.5, with CUDA version 11.1 and CUDNN version 8.1.0.* 

**Installing TF in a Conda Environment**

You can install your own TF instance following these simple steps:

* Load required software modules, e.g.,

<pre>
$ module load python/3.8.5-fasrc01
$ module load cuda/11.1.0-fasrc01
$ module load cudnn/8.1.0.77_cuda11.2-fasrc01
</pre>

* Create a new *conda* environment with Python and some additional packages needed by TensorFlow 
(**Note:** the specific example includes additional packages, such as <code>scipy</code>, <code>pandas</code>, <code>matplotlib</code>, <code>seaborn</code>, <code>h5py</code> and <code>jupyterlab</code>, required for data analytics and visualization.)

<pre>
$ conda create -n tf2.5_cuda11 python=3.8 pip numpy six wheel scipy pandas matplotlib seaborn h5py jupyterlab
</pre>

* Activate the new *conda* environment, e.g.,

<pre>
$ source activate tf2.5_cuda11
(tf2.5_cuda11) $
</pre>

* Install TF with <code>pip</code>, e.g.,

<pre>
(tf2.5_cuda11) $ pip install --upgrade tensorflow-gpu==2.5
</pre>

**Pull a TF singularity container**

Alternatively, one can pull and use a TensorFlow [singularity](https://sylabs.io/guides/3.4/user-guide/index.html) container:

<pre>
$ singularity pull --name tf20_gpu.simg docker://tensorflow/tensorflow:latest-gpu
</pre>

This will result in the image <code>tf20_gpu.simg</code>. The image then can be used with, e.g.,

<pre>
$ singularity exec --nv tf25_gpu.simg python
Python 3.6.9 (default, Jan 26 2021, 15:33:00)
[GCC 8.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf
2021-05-24 21:03:29.150885: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcudart.so.11.0
>>> print(tf.__version__)
2.5.0
</pre>

**Note:** Please notice the use of the <code>--nv</code> option. This is required to make use of the NVIDIA GPU card on the host system.


#### CPU Version

Similarly to the GPU installation you can either install TF in a *conda* environment or use a TF singularity container.

**Installing TF in a Conda Environment**

<pre>
# (1) Load required software modules
$ module load python/3.8.5-fasrc01

# (2) Create conda environment
$ conda create -n tf2.5_cpu python=3.8 pip numpy six wheel scipy pandas matplotlib seaborn h5py jupyterlab

# (3) Activate the conda environment
$ source activate tf2.5_cpu
(tf2.5_cpu)

# (4) Install TF with pip
pip install --upgrade tensorflow==2.5 
</pre>

**Pull a TF singularity container**

<pre>
$ singularity pull --name tf25_cpu.simg docker://tensorflow/tensorflow:latest
</pre>

This will result in the image <code>tf25_cpu.simg</code>. The image then can be used with, e.g.,

<pre>
$ singularity exec --nv tf25_cpu.simg python
Python 3.6.9 (default, Jan 26 2021, 15:33:00)
[GCC 8.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf
>>> print(tf.__version__)
2.5.0
</pre>

### Running TensorFlow:

#### Run TensorFlow interactively

For an **interactive session** to work with the GPUs you can use following:

<pre>
$ salloc -p gpu_test -t 0-06:00 --mem=8000 --gres=gpu:1
</pre>

While on GPU node, you can run <code>nvidia-smi</code> to get information about the assigned GPU's.

<pre>
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
</pre>

Load required modules, and source your TF environment:

<pre>
[username@holygpu2c0701 ~]$ module load python/3.8.5-fasrc01 cuda/11.1.0-fasrc01 cudnn/8.1.0.77_cuda11.2-fasrc01 && source activate tf2.5_cuda11 
(tf2.5_cuda11) [username@holygpu2c0701 ~]$ 
</pre>

Test TF:

(Example adapted from [here](https://www.tensorflow.org/tutorials/keras/classification/).)

<pre>
(tf2.5_cuda11) [username@holygpu2c0701 ~]$ python tf_test.py
2021-05-24 22:10:09.949014: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcudart.so.11.0
2.5.0
2021-05-24 22:10:52.158613: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcuda.so.1
2021-05-24 22:10:52.218295: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1733] Found device 0 with properties:
pciBusID: 0000:06:00.0 name: NVIDIA Tesla V100-PCIE-32GB computeCapability: 7.0
coreClock: 1.38GHz coreCount: 80 deviceMemorySize: 31.75GiB deviceMemoryBandwidth: 836.37GiB/s
2021-05-24 22:10:52.218519: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcudart.so.11.0
2021-05-24 22:10:52.328370: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcublas.so.11
2021-05-24 22:10:52.328721: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcublasLt.so.11
2021-05-24 22:10:52.435103: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcufft.so.10
2021-05-24 22:10:52.543469: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcurand.so.10
2021-05-24 22:10:52.791080: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcusolver.so.11
2021-05-24 22:10:52.931380: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcusparse.so.11
2021-05-24 22:10:52.945169: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcudnn.so.8
2021-05-24 22:10:52.953235: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1871] Adding visible gpu devices: 0
2021-05-24 22:10:52.954007: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2021-05-24 22:10:52.959883: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1733] Found device 0 with properties:
pciBusID: 0000:06:00.0 name: NVIDIA Tesla V100-PCIE-32GB computeCapability: 7.0
coreClock: 1.38GHz coreCount: 80 deviceMemorySize: 31.75GiB deviceMemoryBandwidth: 836.37GiB/s
2021-05-24 22:10:52.964002: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1871] Adding visible gpu devices: 0
2021-05-24 22:10:52.970902: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcudart.so.11.0
2021-05-24 22:10:56.507572: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1258] Device interconnect StreamExecutor with strength 1 edge matrix:
2021-05-24 22:10:56.507720: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1264]      0
2021-05-24 22:10:56.507782: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1277] 0:   N
2021-05-24 22:10:56.511550: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1418] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 30995 MB memory) -> physical GPU (device: 0, name: NVIDIA Tesla V100-PCIE-32GB, pci bus id: 0000:06:00.0, compute capability: 7.0)
2021-05-24 22:11:01.412251: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:176] None of the MLIR Optimization Passes are enabled (registered 2)
2021-05-24 22:11:01.413303: I tensorflow/core/platform/profile_utils/cpu_utils.cc:114] CPU Frequency: 2600000000 Hz
Epoch 1/10
2021-05-24 22:11:02.095831: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcublas.so.11
2021-05-24 22:11:03.831497: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcublasLt.so.11
1875/1875 [==============================] - 5s 1ms/step - loss: 0.4943 - accuracy: 0.8282
Epoch 2/10
1875/1875 [==============================] - 2s 1ms/step - loss: 0.3755 - accuracy: 0.8664
Epoch 3/10
1875/1875 [==============================] - 2s 1ms/step - loss: 0.3366 - accuracy: 0.8766
Epoch 4/10
1875/1875 [==============================] - 2s 1ms/step - loss: 0.3132 - accuracy: 0.8841
Epoch 5/10
1875/1875 [==============================] - 2s 1ms/step - loss: 0.2964 - accuracy: 0.8910
Epoch 6/10
1875/1875 [==============================] - 2s 1ms/step - loss: 0.2806 - accuracy: 0.8962
Epoch 7/10
1875/1875 [==============================] - 2s 1ms/step - loss: 0.2677 - accuracy: 0.9009
Epoch 8/10
1875/1875 [==============================] - 2s 1ms/step - loss: 0.2573 - accuracy: 0.9039
Epoch 9/10
1875/1875 [==============================] - 2s 1ms/step - loss: 0.2458 - accuracy: 0.9069
Epoch 10/10
1875/1875 [==============================] - 2s 1ms/step - loss: 0.2375 - accuracy: 0.9112
313/313 - 0s - loss: 0.3409 - accuracy: 0.8789

Test accuracy: 0.8788999915122986
[1.3856260e-08 8.9077723e-10 1.6243800e-11 2.3399921e-10 2.4639154e-12
 1.9129384e-05 1.4301588e-08 2.6651691e-03 7.0685728e-09 9.9731570e-01]
9
(tf2.5_cuda11) [username@holygpu2c0701 ~]$
</pre>

In the above example we used the following test code, <code>tf_test.py</code>:

<pre>
#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals
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
</pre>


#### Examples:

* [Example 1](Example1): Simple 2D CNN with the MNIST dataset

### Suggested Reading:

* [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
* [TensorFlow API](https://www.tensorflow.org/api_docs/python)
