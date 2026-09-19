import tensorflow as tf
import numpy as np

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

print("Before preprocessing:")
print("Minimum pixel value:", x_train.min())
print("Maximum pixel value:", x_train.max())

# Normalize pixel values from 0-255 to 0-1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print("\nAfter preprocessing:")
print("Minimum pixel value:", x_train.min())
print("Maximum pixel value:", x_train.max())

print("\nTraining image shape:", x_train.shape)
print("Testing image shape:", x_test.shape)