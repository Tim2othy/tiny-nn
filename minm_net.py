import struct
from array import array

import numpy as np

LR = 0.04


def get_data():

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

    # Preprocess the training data
    # Reshape to (num_samples, 1, 28*28) and normalize to range [0, 1]
    x_train = x_train.reshape(x_train.shape[0], 1, 28 * 28).astype("float32") / 255
    x_test = x_test.reshape(x_test.shape[0], 1, 28 * 28).astype("float32") / 255
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    training_data = (x_train, y_train)
    test_data = (x_test, y_test)

    return training_data, test_data


"""Our Loss function and its derivative."""


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size


""" Forwardpass for our activation functions and fully connected layer """


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    return probabilities


def fc_fp(bias, weights, input):
    return np.dot(input, weights) + bias


"""Backwardpass for relu and fully connected layer"""


def relu_prime(x):
    return np.where(x > 0, 1, 0)


# Returns input_error=dE/dX for a given output_error=dE/dY.
def relu_bp(input, output_error):
    return relu_prime(input) * output_error


# computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
def fc_bp(bias, weights, input, output_error):
    input_error = np.dot(output_error, weights.T)
    weights_error = np.dot(input.T, output_error)

    # update parameters
    weights -= LR * weights_error
    bias -= LR * output_error
    return input_error


"""Network functions"""


def train(data):
    samples = len(data[0])
    loss_print = 0

    # training loop
    for i in range(samples):
        """forward propagation"""
        pixels = data[0][i]

        z1 = fc_fp(b1, w1, pixels)
        activation1 = relu(z1)

        z2 = fc_fp(b2, w2, activation1)
        activation2 = relu(z2)

        logit = fc_fp(b3, w3, activation2)
        prediction = softmax(logit)

        # compute loss (for display purpose only)
        loss_print += mse(data[1][i], prediction)

        """backward propagation"""
        error = mse_prime(data[1][i], prediction)

        # we can skip softmax
        error = fc_bp(b3, w3, activation2, error)
        error = relu_bp(z2, error)
        error = fc_bp(b2, w2, activation1, error)
        error = relu_bp(z1, error)
        error = fc_bp(b1, w1, pixels, error)

        if (i + 1) % samples // 10 == 0:
            loss_print /= samples // 10

            print(f"For the sample {i + 1}/{samples}   the error is {loss_print}")
            loss_print = 0


def evaluate(data):
    samples = len(data[0])
    loss_print = 0

    # run network over all samples
    for i in range(samples):
        """forward propagation"""
        pixels = data[0][i]

        output = fc_fp(b1, w1, pixels)
        output = relu(output)
        output = fc_fp(b2, w2, output)
        output = relu(output)
        logit = fc_fp(b3, w3, output)
        prediction = softmax(logit)

        loss_print += mse(data[1][i], prediction)

    print("Test loss:", loss_print / samples)


"""Creating Neural Network"""


# Create matrices for learnable parameters
w1 = np.random.rand(28 * 28, 100) - 0.5
b1 = np.random.rand(1, 100) - 0.5
w2 = np.random.rand(100, 50) - 0.5
b2 = np.random.rand(1, 50) - 0.5
w3 = np.random.rand(50, 10) - 0.5
b3 = np.random.rand(1, 10) - 0.5

training_data, test_data = get_data()

training_data = (training_data[0][:5000], training_data[1][:5000])

# train the network
train(training_data)

# evaluate on test data
evaluate(test_data)
