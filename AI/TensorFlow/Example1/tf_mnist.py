#!/usr/bin/env python
from __future__ import print_function
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
epochs = 20
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
