import struct
from array import array

import numpy as np


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

    return x_train, y_train, x_test, y_test


LEARNING_RATE = 0.04
EPOCHS = 20

#
# Layers
#


def fc_fp(bias, weights, input):
    input = input
    output = (
        np.dot(input, weights) + bias
    )  # So this is the output(input) i gues the dot product of the input and weights + the bias, makes sense
    return output


# computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
def fc_bp(bias, weights, input, output_error):
    input_error = np.dot(output_error, weights.T)
    weights_error = np.dot(input.T, output_error)
    # dBias = output_error

    # update parameters
    weights -= (
        LEARNING_RATE * weights_error
    )  # Makes sense the weight becomes itself minus the learning weigt times weight error I guess this is nabla E or so
    bias -= LEARNING_RATE * output_error
    return input_error


def softmax_fp(input):
    # Compute the softmax output
    exp_values = np.exp(input - np.max(input, axis=1, keepdims=True))
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    return probabilities


def relu(x):
    return np.maximum(0, x)


def relu_prime(x):
    return np.where(x > 0, 1, 0)


def relu_fp(input):
    output = relu(input)
    return output


# Returns input_error=dE/dX for a given output_error=dE/dY.
def relu_bp(input, output_error):
    return relu_prime(input) * output_error


# loss function and its derivative
def mse(y_true, y_pred):

    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size


#
# network
#
def fit(x_train, y_train):
    # sample dimension first
    samples = len(x_train)

    # training loop
    for i in range(EPOCHS):
        err = 0
        for j in range(samples):

            # forward propagation
            pixels = x_train[j]

            z1 = fc_fp(b1, w1, pixels)
            activation1 = relu_fp(z1)

            z2 = fc_fp(b2, w2, activation1)
            activation2 = relu_fp(z2)

            z3 = fc_fp(b3, w3, activation2)
            output = softmax_fp(z3)

            # compute loss (for display purpose only)
            err = err + mse(y_train[j], output)

            # backward propagation
            error = mse_prime(y_train[j], output)

            # skipping softmax since error isn't changed
            error = fc_bp(b3, w3, activation2, error)
            error = relu_bp(z2, error)
            error = fc_bp(b2, w2, activation1, error)
            error = relu_bp(z1, error)
            error = fc_bp(b1, w1, pixels, error)

        # calculate average error on all samples
        err /= samples

        if (i + 1) % round(EPOCHS / 10) == 0 or i == 0:
            print("For the epoch %d/%d   the error is %f" % (i + 1, EPOCHS, err))


# predict output for given input
def predict(input_data):
    # sample dimension first
    samples = len(input_data)
    result = []

    # run network over all samples
    for i in range(samples):
        # forward propagation
        output = input_data[i]

        # Manual forward pass
        output = fc_fp(b1, w1, output)
        output = relu_fp(output)
        output = fc_fp(b2, w2, output)
        output = relu_fp(output)
        output = fc_fp(b3, w3, output)
        output = softmax_fp(output)

        result.append(output)

    return result


def evaluate(x_test, y_test):
    samples = len(x_test)
    err = 0

    prediction = predict(x_test)

    # run network over all samples
    for i in range(samples):

        err_i = mse(y_test[i], prediction[i])

        err = err + err_i

    return err / samples


# Create matrices for learnable parameters
w1 = np.random.rand(28 * 28, 100) - 0.5
b1 = np.random.rand(1, 100) - 0.5
w2 = np.random.rand(100, 50) - 0.5
b2 = np.random.rand(1, 50) - 0.5
w3 = np.random.rand(50, 10) - 0.5
b3 = np.random.rand(1, 10) - 0.5

x_train, y_train, x_test, y_test = get_data()

# train the network
fit(x_train[:4000], y_train[:4000])

# evaluate on test data
test_loss = evaluate(x_test[:100], y_test[:100])
print("Test loss:", test_loss)
