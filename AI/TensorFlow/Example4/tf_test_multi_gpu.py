#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
import numpy as np

print(tf.__version__)

# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

# Open a strategy scope.
with strategy.scope():
    # Everything that creates variables should be under the strategy scope.
    # In general this is only model construction & `compile()`.
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

# you can change verbose=1 to see progress bars when running interactively
model.fit(train_images, train_labels, epochs=10, verbose=2)

# you can change verbose=1 to see progress bars when running interactively
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# you can change verbose=1 to see progress bars when running interactively
predictions = model.predict(test_images, verbose=2)
print(predictions[0])
print(np.argmax(predictions[0]))


