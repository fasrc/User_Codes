## TensorFlow
<img src="Images/tensorflow-logo.png" alt="TF-logo" width="200"/>

### What is TensorFlow?

[TensorFlow](https://www.tensorflow.org) (TF) is an open-source software library for numerical computation using data flow graphs. Nodes in the graph represent mathematical operations, while the graph edges represent the multidimensional data arrays (tensors) that flow between them. This flexible architecture lets you deploy computation to one or more CPUs or GPUs in a desktop, server, or mobile device without rewriting code.

TensorFlow was originally developed by researchers and engineers working on the Google Brain team within Google's Machine Intelligence research organization for the purposes of conducting machine learning and deep neural networks research. The system is general enough to be applicable in a wide variety of other domains, as well.

### Installing TensorFlow:

The below instructions are intended to help you set up TF on the FASRC cluster.

#### GPU Version

The specific example illustrates the installation of TF version 2.0 with Python version 3.7, CUDA version 10.0, and CUDNN version 7.4. Please refer to our documentation on [running GPU jobs on the FASRC cluster](https://www.rc.fas.harvard.edu/resources/documentation/gpgpu-computing-on-the-cluster/).

The two recommended methods for setting up TF in your user environment is installing TF in a [conda environment](https://www.rc.fas.harvard.edu/resources/documentation/software-on-the-cluster/python/) in your user space, or use a TF [singularity container](https://www.rc.fas.harvard.edu/resources/documentation/software/singularity-on-the-cluster). 

**Installing TF in a Conda Environment**

You can install your own TF instance following these simple steps:

* Load required software modules, e.g.,

```bash
$ module load python/3.6.3-fasrc02
$ module load cuda/10.1.243-fasrc01
$ module load cudnn/7.6.5.32_cuda10.1-fasrc01
```

* Create a new *conda* environment with Python and some additional packages needed by TensorFlow 
(**Note:** the specific example includes additional packages, such as <code>scipy</code>, <code>pandas</code>, <code>matplotlib</code>, <code>seaborn</code>, <code>h5py</code> and <code>jupyterlab</code>, required for data analytics and visualization.)

```bash
$ conda create -n tf2.0_cuda10 python=3.7 pip numpy six wheel scipy pandas matplotlib seaborn h5py jupyterlab
```

* Activate the new *conda* environment, e.g.,

```bash
$ source activate tf2.0_cuda10
(tf2.0_cuda10) $
```

* Install TF with <code>pip</code>, e.g.,

```bash
(tf2.0_cuda10) $ pip install --upgrade tensorflow-gpu==2.0
...
Installing collected packages: absl-py, werkzeug, grpcio, pyasn1, pyasn1-modules, rsa, cachetools, google-auth, protobuf, markdown, oauthlib, idna, urllib3, chardet, requests, requests-oauthlib, google-auth-oauthlib, tensorboard, tensorflow-estimator, termcolor, keras-applications, gast, google-pasta, keras-preprocessing, opt-einsum, wrapt, astor, tensorflow-gpu
Successfully installed absl-py-0.8.1 astor-0.8.0 cachetools-3.1.1 chardet-3.0.4 gast-0.2.2 google-auth-1.6.3 google-auth-oauthlib-0.4.1 google-pasta-0.1.8 grpcio-1.24.3 idna-2.8 keras-applications-1.0.8 keras-preprocessing-1.1.0 markdown-3.1.1 oauthlib-3.1.0 opt-einsum-3.1.0 protobuf-3.10.0 pyasn1-0.4.7 pyasn1-modules-0.2.7 requests-2.22.0 requests-oauthlib-1.2.0 rsa-4.0 tensorboard-2.0.1 tensorflow-estimator-2.0.1 tensorflow-gpu-2.0.0 termcolor-1.1.0 urllib3-1.25.6 werkzeug-0.16.0 wrapt-1.11.2
```

**Pull a TF singularity container**

Alternatively, one can pull and use a TensorFlow [singularity](https://sylabs.io/guides/3.4/user-guide/index.html) container:

```bash
$ singularity pull --name tf20_gpu.simg docker://tensorflow/tensorflow:latest-gpu
```

This will result in the image <code>tf20_gpu.simg</code>. The image then can be used with, e.g.,

```bash
$ singularity exec --nv tf20_gpu.simg python
Python 2.7.15+ (default, Jul  9 2019, 16:51:35) 
[GCC 7.4.0] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>>
```

**Note:** Please notice the use of the <code>--nv</code> option. This is required to make use of the NVIDIA GPU card on the host system.


#### CPU Version

Similarly to the GPU installation you can either install TF in a *conda* environment or use a TF singularity container.

**Installing TF in a Conda Environment**

```bash
# (1) Load required software modules
$ module load python/3.6.3-fasrc02

# (2) Create conda environment
$ conda create -n tf2.0_cpu python=3.7 pip numpy six wheel scipy pandas matplotlib seaborn h5py jupyterlab

# (3) Activate the conda environment
$ source activate tf2.0_cpu
(tf2.0_cpu)

# (4) Install TF with pip
pip install --upgrade tensorflow==2.0 
```

**Pull a TF singularity container**

```bash
$ singularity pull --name tf20_cpu.simg docker://tensorflow/tensorflow:latest
```

This will result in the image <code>tf20_cpu.simg</code>. The image then can be used with, e.g.,

```bash
$ singularity exec tf20_cpu.simg python
Python 2.7.15+ (default, Jul  9 2019, 16:51:35) 
[GCC 7.4.0] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf
>>> print (tf.__version__)
2.0.0
```

### Running TensorFlow:

#### Run TensorFlow interactively

For an **interactive session** to work with the GPUs you can use following:

```bash
$ srun --pty -p gpu -t 0-06:00 --mem=8000 --gres=gpu:1 /bin/bash
```

While on GPU node, you can run <code>nvidia-smi</code> to get information about the assigned GPU's.

```bash
[username@holygpu2c0716 ~]$ nvidia-smi
Mon Nov  4 18:18:14 2019       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 418.87.00    Driver Version: 418.87.00    CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla V100-PCIE...  On   | 00000000:D8:00.0 Off |                    0 |
| N/A   38C    P0    28W / 250W |      0MiB / 32480MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

Load required modules, and source your TF environment:

```bash
[username@holygpu2c0716 ~]$ module load cuda/10.1.243-fasrc01 cudnn/7.6.5.32_cuda10.1-fasrc01 python/3.6.3-fasrc02 && source activate tf2.0_cuda10 
(tf2.0_cuda10) [username@holygpu2c0716 ~]$ 
```

Test TF:

(Example adapted from [here](https://www.tensorflow.org/tutorials/keras/classification/).)

```
(tf2.0_cuda10) [username@holygpu2c0716 ~]$ python
Python 3.7.5 (default, Oct 25 2019, 15:51:11) 
[GCC 7.3.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from __future__ import absolute_import, division, print_function, unicode_literals
>>> import tensorflow as tf
>>> from tensorflow import keras
>>> import numpy as np
>>> print(tf.__version__)
2.0.0
>>> fashion_mnist = keras.datasets.fashion_mnist
>>> (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
>>> class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
...                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
>>> train_images.shape
(60000, 28, 28)
>>> len(train_labels)
60000
>>> train_labels
array([9, 0, 0, ..., 3, 0, 5], dtype=uint8)
>>> train_images = train_images / 255.0
>>> test_images = test_images / 255.0
>>> model = keras.Sequential([
...     keras.layers.Flatten(input_shape=(28, 28)),
...     keras.layers.Dense(128, activation='relu'),
...     keras.layers.Dense(10, activation='softmax')
... ])
2019-11-04 18:29:29.266481: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
2019-11-04 18:29:29.381100: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties: 
name: Tesla V100-PCIE-32GB major: 7 minor: 0 memoryClockRate(GHz): 1.38
pciBusID: 0000:d8:00.0
2019-11-04 18:29:29.397063: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
2019-11-04 18:29:29.414381: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
2019-11-04 18:29:29.433180: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10.0
2019-11-04 18:29:29.449636: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10.0
2019-11-04 18:29:29.470325: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10.0
2019-11-04 18:29:29.489725: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10.0
2019-11-04 18:29:29.516071: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2019-11-04 18:29:29.520270: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
2019-11-04 18:29:29.520765: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2019-11-04 18:29:29.530225: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2600000000 Hz
2019-11-04 18:29:29.530379: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x559b751032f0 executing computations on platform Host. Devices:
2019-11-04 18:29:29.530430: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): Host, Default Version
2019-11-04 18:29:29.713705: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x559b75105c70 executing computations on platform CUDA. Devices:
2019-11-04 18:29:29.713835: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): Tesla V100-PCIE-32GB, Compute Capability 7.0
2019-11-04 18:29:29.715455: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties: 
name: Tesla V100-PCIE-32GB major: 7 minor: 0 memoryClockRate(GHz): 1.38
pciBusID: 0000:d8:00.0
2019-11-04 18:29:29.715639: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
2019-11-04 18:29:29.715692: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
2019-11-04 18:29:29.715765: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10.0
2019-11-04 18:29:29.715817: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10.0
2019-11-04 18:29:29.715869: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10.0
2019-11-04 18:29:29.715922: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10.0
2019-11-04 18:29:29.715974: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2019-11-04 18:29:29.722615: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
2019-11-04 18:29:29.722716: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
2019-11-04 18:29:29.724692: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1159] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-11-04 18:29:29.724771: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1165]      0 
2019-11-04 18:29:29.724817: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1178] 0:   N 
2019-11-04 18:29:29.727810: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1304] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 30555 MB memory) -> physical GPU (device: 0, name: Tesla V100-PCIE-32GB, pci bus id: 0000:d8:00.0, compute capability: 7.0)
>>> model.compile(optimizer='adam',
...               loss='sparse_categorical_crossentropy',
...               metrics=['accuracy'])
>>> model.fit(train_images, train_labels, epochs=10)
Train on 60000 samples
Epoch 1/10
2019-11-04 18:30:09.921767: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
60000/60000 [==============================] - 5s 90us/sample - loss: 0.4990 - accuracy: 0.8246
Epoch 2/10
60000/60000 [==============================] - 3s 46us/sample - loss: 0.3770 - accuracy: 0.8640
Epoch 3/10
60000/60000 [==============================] - 3s 46us/sample - loss: 0.3371 - accuracy: 0.8776
Epoch 4/10
60000/60000 [==============================] - 3s 46us/sample - loss: 0.3149 - accuracy: 0.8836
Epoch 5/10
60000/60000 [==============================] - 3s 46us/sample - loss: 0.2987 - accuracy: 0.8899
Epoch 6/10
60000/60000 [==============================] - 3s 46us/sample - loss: 0.2831 - accuracy: 0.8949
Epoch 7/10
60000/60000 [==============================] - 3s 47us/sample - loss: 0.2708 - accuracy: 0.9001
Epoch 8/10
60000/60000 [==============================] - 3s 46us/sample - loss: 0.2576 - accuracy: 0.9034
Epoch 9/10
60000/60000 [==============================] - 3s 46us/sample - loss: 0.2502 - accuracy: 0.9069
Epoch 10/10
60000/60000 [==============================] - 3s 46us/sample - loss: 0.2406 - accuracy: 0.9094
<tensorflow.python.keras.callbacks.History object at 0x2b35a3880bd0>
>>> test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
10000/1 - 1s - loss: 0.2565 - accuracy: 0.8850
>>> print('\nTest accuracy:', test_acc)

Test accuracy: 0.885
>>> predictions = model.predict(test_images)
>>> predictions[0]
array([7.2989198e-10, 2.9799860e-11, 1.2714995e-09, 9.6692572e-14,
       8.5295845e-08, 1.5101463e-03, 4.9370102e-10, 5.2422038e-03,
       3.9883501e-09, 9.9324751e-01], dtype=float32)
>>> np.argmax(predictions[0])
9
>>> quit()
(tf2.0_cuda10) [username@holygpu2c0716 ~]$
```

### Suggested Reading:

* [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
* [TensorFlow API](https://www.tensorflow.org/api_docs/python)
