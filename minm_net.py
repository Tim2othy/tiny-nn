import struct
from array import array

import numpy as np


# loss function and its derivative
def mse(y_true, y_pred):

    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size


# Base class
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    # computes the output Y of a layer for a given input X - We can always know what X will be for some Y
    def forward_propagation(self, input):
        raise NotImplementedError

    # computes dE/dX for a given dE/dY (and update parameters if any)
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError

    #  We can find out how E changes with a small change in Y easily
    # Using the chain rule we find out how a change in the input would change E


# inherit from base class Layer
class FCLayer(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size):
        self.weights = (
            np.random.rand(input_size, output_size) - 0.5
        )  # So the weights and biases are randomized
        self.bias = np.random.rand(1, output_size) - 0.5

    # returns output for a given input
    def forward_propagation(self, input):
        self.input = input
        self.output = (
            np.dot(self.input, self.weights) + self.bias
        )  # So this is the output(input) i gues the dot product of the input and weights + the bias, makes sense
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # dBias = output_error

        # update parameters
        self.weights -= (
            learning_rate * weights_error
        )  # Makes sense the weight becomes itself minus the learning weigt times weight error I guess this is nabla E or so
        self.bias -= learning_rate * output_error
        return input_error


# activation function and its derivative
def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1 - np.tanh(x) ** 2


def relu(x):
    return np.maximum(0, x)


def relu_prime(x):
    return np.where(x > 0, 1, 0)


class SoftmaxLayer(Layer):
    def forward_propagation(self, input):
        # Compute the softmax output
        exp_values = np.exp(input - np.max(input, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities

    def backward_propagation(self, output_error, learning_rate):
        # Compute the gradient of the softmax function
        return (
            output_error  # No gradient update needed for softmax layer in this context
        )


# inherit from base class Layer
class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    # returns the activated input
    def forward_propagation(self, input):
        self.input = input
        self.output = self.activation(self.input)
        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there are no "learnable" parameters.
    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error


class Network:
    def __init__(self, loss, loss_prime):
        self.layers = []
        self.loss = loss
        self.loss_prime = loss_prime

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    # evaluate results for some data
    def evaluate(self, x_test, y_test):
        # sample dimension first
        samples = len(x_test)
        err = 0

        prediction = self.predict(x_test)

        # run network over all samples
        for i in range(samples):

            example_error = self.loss(y_test[i], prediction[i])

            err = err + example_error

        return err / samples

    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate):
        # Ensure loss function is set
        if self.loss is None:
            raise ValueError(
                "Loss function is not set. Use the 'use' method to set the loss and loss_prime."
            )
        if self.loss_prime is None:
            raise ValueError(
                "Loss function is not set. Use the 'use' method to set the loss and loss_prime."
            )
        # sample dimension first
        samples = len(x_train)

        # training loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display purpose only)
                err = err + self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # calculate average error on all samples
            err /= samples

            if (i + 1) % round(epochs / 10) == 0 or i == 0:
                print("For the epoch %d/%d   the error is %f" % (i + 1, epochs, err))


# dict of file paths
paths = {
    "train_img": "input/train-images-idx3-ubyte",
    "train_lab": "input/train-labels-idx1-ubyte",
    "test_img": "input/t10k-images-idx3-ubyte",
    "test_lab": "input/t10k-labels-idx1-ubyte",
}


def to_categorical(y, num_classes=10):
    """Convert class vector to binary class matrix (one-hot encoding)"""
    y = np.array(y, dtype="int")
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical


# MNIST Data Loader Class


class MnistDataloader(object):
    def __init__(
        self,
        training_images_filepath,
        training_labels_filepath,
        test_images_filepath,
        test_labels_filepath,
    ):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, "rb") as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(
                    "Magic number mismatch, expected 2049, got {}".format(magic)
                )
            labels = array("B", file.read())

        with open(images_filepath, "rb") as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(
                    "Magic number mismatch, expected 2051, got {}".format(magic)
                )
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols : (i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(
            self.training_images_filepath, self.training_labels_filepath
        )
        x_test, y_test = self.read_images_labels(
            self.test_images_filepath, self.test_labels_filepath
        )
        return (x_train, y_train), (x_test, y_test)


mnist_dataloader = MnistDataloader(
    paths["train_img"],
    paths["train_lab"],
    paths["test_img"],
    paths["test_lab"],
)

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()


# Convert to numpy arrays first
x_train = np.array(x_train, dtype=np.float32)
x_test = np.array(x_test, dtype=np.float32)


# Preprocess the training data
# Reshape to (num_samples, 1, 28*28) and normalize to range [0, 1]
x_train = x_train.reshape(x_train.shape[0], 1, 28 * 28).astype("float32") / 255
# Convert labels to one-hot encoding
y_train = to_categorical(y_train)

# Preprocess the test data
x_test = x_test.reshape(x_test.shape[0], 1, 28 * 28).astype("float32") / 255
y_test = to_categorical(y_test)

# Create the network
net = Network(mse, mse_prime)
net.add(FCLayer(28 * 28, 100))  # input_shape=(1, 28*28)    ;   output_shape=(1, 100)
net.add(ActivationLayer(relu, relu_prime))
net.add(FCLayer(100, 50))  # input_shape=(1, 100)      ;   output_shape=(1, 50)
net.add(ActivationLayer(relu, relu_prime))
net.add(FCLayer(50, 10))  # input_shape=(1, 50)       ;   output_shape=(1, 10)
net.add(SoftmaxLayer())

# train the network
net.fit(x_train[:4000], y_train[:4000], epochs=12, learning_rate=0.04)

# evaluate on test data
test_loss = net.evaluate(x_test[:100], y_test[:100])
print("Test loss:", test_loss)

# visualize the network on a few samples
# out = net.predict(x_test[:8])
# print("\nPredicted values:")
# print(np.round(out, 1))
# print("True values:")
# print(y_test[:8])
