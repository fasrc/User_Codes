### Purpose:
Simple 2D CNN with the MNIST dataset. Example also illustrates computing the confusion matrix, and ROC curves for each class. 

### Contents:

* <code>tf_mnist.py</code>: Python source code
* <code>run.sbatch</code>: Batch-job submission script
* <code>tf_mnist.out</code>: Output file
* <code>tf_example1.ipynb</code>: Jupyter notebook

### Python source code:

```python
#!/usr/bin/env python
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow.keras as K

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy.interpolate import interp1d

def ROC_curves(y_actu, pred, classes):
    """Computes ROC curves for each class"""

    yt = label_binarize(y_actu, np.arange(classes))
    n_classes = yt.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in np.arange(n_classes):
        fpr[i], tpr[i], _ = roc_curve(yt[:, i], pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    return fpr, tpr, roc_auc

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# --- Input image dimensions ---
img_rows, img_cols = 28, 28

if K.backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# --- Model ---
num_classes = 10
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

# --- Build model ---
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# --- Train model ---
batch_size = 512
epochs = 10
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

# --- Evaluate model ---
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# --- Predictions ---
predictions = model.predict(x_test)
N = len(x_test)
y_predicted = np.zeros(N)
for i in np.arange(0, N):
    predictions_array = predictions[i,:]
    predicted_label = np.argmax(predictions_array)
    y_predicted[i] = int(predicted_label)

# --- Confusion matrix ---
y_actu = y_test.astype(int)
y_pred = y_predicted.astype(int)
cm = confusion_matrix(y_actu, y_pred)
print('Confusion Matrix:')
print(cm)

# --- ROC curves for each class ---
FPR, TPR, AUC = ROC_curves(y_actu, predictions, num_classes)
print('')
print('--- ROC curves ---')
print('True Positive Rate for class 0:')
print(TPR[0])
print('False Positive Rate: for class 0')
print(FPR[0])
print('AUC = %0.3f' % AUC[0])
```

### Example batch-job submission script:

```bash
#!/bin/bash
#SBATCH -p gpu_test
#SBATCH -n 1
#SBATCH -t 0-03:00
#SBATCH -J dnn
#SBATCH -o tf_mnist.out
#SBATCH -e tf_mnist.err
#SBATCH --gres=gpu:1
#SBATCH --gpu-freq=high
#SBATCH --mem=8G

# --- Set up software environment ---
module load python/3.8.5-fasrc01 
module load cuda/11.7.1-fasrc01
module load cudnn/8.5.0.96_cuda11-fasrc01 
module load gcc/12.1.0-fasrc01
source activate tf2.10_cuda11

# --- Run the code ---
srun -n 1 --gres=gpu:1 python tf_mnist.py 
```

### Example usage:

```bash
sbatch run.sbatch
```

### Example output:

```
cat tf_mnist.out
x_train shape: (60000, 28, 28, 1)
60000 train samples
10000 test samples
Epoch 1/10
118/118 [==============================] - 2s 15ms/step - loss: 0.3234 - accuracy: 0.9047 - val_loss: 0.0771 - val_accuracy: 0.9761
Epoch 2/10
118/118 [==============================] - 2s 13ms/step - loss: 0.0793 - accuracy: 0.9762 - val_loss: 0.0444 - val_accuracy: 0.9851
Epoch 3/10
118/118 [==============================] - 2s 13ms/step - loss: 0.0524 - accuracy: 0.9842 - val_loss: 0.0356 - val_accuracy: 0.9876
Epoch 4/10
118/118 [==============================] - 2s 13ms/step - loss: 0.0399 - accuracy: 0.9875 - val_loss: 0.0293 - val_accuracy: 0.9900
Epoch 5/10
118/118 [==============================] - 2s 13ms/step - loss: 0.0306 - accuracy: 0.9907 - val_loss: 0.0290 - val_accuracy: 0.9902
Epoch 6/10
118/118 [==============================] - 2s 13ms/step - loss: 0.0269 - accuracy: 0.9916 - val_loss: 0.0288 - val_accuracy: 0.9912
Epoch 7/10
118/118 [==============================] - 2s 13ms/step - loss: 0.0218 - accuracy: 0.9930 - val_loss: 0.0249 - val_accuracy: 0.9916
Epoch 8/10
118/118 [==============================] - 2s 13ms/step - loss: 0.0195 - accuracy: 0.9938 - val_loss: 0.0282 - val_accuracy: 0.9914
Epoch 9/10
118/118 [==============================] - 2s 13ms/step - loss: 0.0158 - accuracy: 0.9947 - val_loss: 0.0260 - val_accuracy: 0.9921
Epoch 10/10
118/118 [==============================] - 2s 13ms/step - loss: 0.0135 - accuracy: 0.9958 - val_loss: 0.0318 - val_accuracy: 0.9906
Test loss: 0.03178912401199341
Test accuracy: 0.9905999898910522
Confusion Matrix:
[[ 978    0    0    0    0    1    0    0    0    1]
 [   0 1131    1    0    0    0    1    0    1    1]
 [   1    0 1027    0    0    0    0    4    0    0]
 [   0    0    0 1003    0    5    0    0    2    0]
 [   0    0    0    0  963    0    4    0    1   14]
 [   1    0    0    8    0  882    1    0    0    0]
 [   7    2    1    0    1    1  945    0    1    0]
 [   0    0    5    2    0    0    0 1016    1    4]
 [   4    0    1    1    0    1    0    2  963    2]
 [   1    0    0    1    2    3    0    2    2  998]]

--- ROC curves ---
True Positive Rate for class 0:
[0.         0.61326531 0.69081633 0.71938776 0.74183673 0.75612245
 0.76938776 0.78061224 0.78877551 0.79387755 0.80612245 0.81428571
 0.82040816 0.8244898  0.83367347 0.83877551 0.84081633 0.84183673
 0.85       0.85306122 0.85510204 0.85612245 0.86020408 0.86326531
 0.86734694 0.86938776 0.87244898 0.8744898  0.8755102  0.87959184
 0.88265306 0.88571429 0.88673469 0.88877551 0.89183673 0.89489796
 0.89897959 0.90102041 0.90510204 0.90714286 0.91020408 0.9122449
 0.91734694 0.91938776 0.92346939 0.9255102  0.93367347 0.93367347
 0.94081633 0.94285714 0.97244898 0.97244898 0.97755102 0.97755102
 0.99183673 0.99183673 0.99489796 0.99489796 0.99591837 0.99591837
 0.99795918 0.99795918 0.99897959 0.99897959 1.         1.        ]
False Positive Rate: for class 0
[0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
 0.00000000e+00 0.00000000e+00 0.00000000e+00 1.10864745e-04
 1.10864745e-04 1.10864745e-04 1.10864745e-04 2.21729490e-04
 2.21729490e-04 3.32594235e-04 3.32594235e-04 5.54323725e-04
 5.54323725e-04 7.76053215e-04 7.76053215e-04 1.21951220e-03
 1.21951220e-03 1.55210643e-03 1.55210643e-03 4.21286031e-03
 4.21286031e-03 1.00000000e+00]
AUC = 1.000
```
