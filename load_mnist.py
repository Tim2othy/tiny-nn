import struct
from array import array

import numpy as np


def to_categorical(y, num_classes=10):
    """Convert class vector to binary class matrix (one-hot encoding)"""
    y = np.array(y, dtype="int")
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical


def read_images_labels(images_filepath, labels_filepath):
    labels = []
    with open(labels_filepath, "rb") as file:
        magic, size = struct.unpack(">II", file.read(8))
        labels = array("B", file.read())

    with open(images_filepath, "rb") as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))

        image_data = array("B", file.read())
    images = []
    for i in range(size):
        images.append([0] * rows * cols)
    for i in range(size):
        img = np.array(image_data[i * rows * cols : (i + 1) * rows * cols])
        img = img.reshape(28, 28)
        images[i][:] = img

    images = np.array(images, dtype=np.float32)

    return images, labels


x_train, y_train = read_images_labels(
    "input/train-images-idx3-ubyte", "input/train-labels-idx1-ubyte"
)

x_test, y_test = read_images_labels(
    "input/t10k-images-idx3-ubyte", "input/t10k-labels-idx1-ubyte"
)

x_train = x_train.reshape(x_train.shape[0], 1, 28 * 28).astype("float32") / 255
x_test = x_test.reshape(x_test.shape[0], 1, 28 * 28).astype("float32") / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

training_data = (x_train, y_train)
test_data = (x_test, y_test)

xt = x_train
xs = x_test
yt = y_train
ys = y_test