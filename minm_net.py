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
def fc_bp(bias, weights, input, output_error, learning_rate):
    input_error = np.dot(output_error, weights.T)
    weights_error = np.dot(input.T, output_error)
    # dBias = output_error

    # update parameters
    weights -= (
        learning_rate * weights_error
    )  # Makes sense the weight becomes itself minus the learning weigt times weight error I guess this is nabla E or so
    bias -= learning_rate * output_error
    return input_error


def softmax_fp(input):
    # Compute the softmax output
    exp_values = np.exp(input - np.max(input, axis=1, keepdims=True))
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    return probabilities


def softmax_bp(output_error):
    # Compute the gradient of the softmax function
    return output_error  # No gradient update needed for softmax layer in this context


def relu(x):
    return np.maximum(0, x)


def relu_prime(x):
    return np.where(x > 0, 1, 0)


def relu_fp(input):
    output = relu(input)
    return output


# Returns input_error=dE/dX for a given output_error=dE/dY.
# learning_rate is not used because there are no "learnable" parameters.
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
def fit(x_train, y_train, epochs, learning_rate):
    # sample dimension first
    samples = len(x_train)

    # training loop
    for i in range(epochs):
        err = 0
        for j in range(samples):
            # forward propagation
            output = x_train[j]

            output = fc_fp(b1, w1, output)
            output = relu_fp(output)
            output = fc_fp(b2, w2, output)
            output = relu_fp(output)
            output = fc_fp(b3, w3, output)
            output = softmax_fp(output)

            # compute loss (for display purpose only)
            err = err + mse(y_train[j], output)

            # backward propagation
            error = mse_prime(y_train[j], output)

            error = softmax_bp(error)
            error = fc_bp(b3, w3, output, error, learning_rate)
            error = relu_bp(output, error)
            error = fc_bp(b2, w2, output, error, learning_rate)
            error = relu_bp(output, error)
            error = fc_bp(b1, w1, output, error, learning_rate)

        # calculate average error on all samples
        err /= samples

        if (i + 1) % round(epochs / 10) == 0 or i == 0:
            print("For the epoch %d/%d   the error is %f" % (i + 1, epochs, err))


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


# evaluate results for some data
def evaluate(x_test, y_test):
    # sample dimension first
    samples = len(x_test)
    err = 0

    prediction = predict(x_test)

    # run network over all samples
    for i in range(samples):

        example_error = mse(y_test[i], prediction[i])

        err = err + example_error

    return err / samples


layers = []

# Initialize weights and biases
w1 = np.random.rand(28 * 28, 100) - 0.5
b1 = np.random.rand(1, 100) - 0.5
w2 = np.random.rand(100, 50) - 0.5
b2 = np.random.rand(1, 50) - 0.5
w3 = np.random.rand(50, 10) - 0.5
b3 = np.random.rand(1, 10) - 0.5


#
# Create the network
#
# layers.append(FCLayer(28 * 28, 100))  # input_shape=(1, 28*28);   output_shape=(1, 100)
# layers.append(ReluLayer())
# layers.append(FCLayer(100, 50))  # input_shape=(1, 100)       ;   output_shape=(1, 50)
# layers.append(ReluLayer())
# layers.append(FCLayer(50, 10))  # input_shape=(1, 50)         ;   output_shape=(1, 10)
# layers.append(SoftmaxLayer())

x_train, y_train, x_test, y_test = get_data()

# train the network
fit(x_train[:4000], y_train[:4000], epochs=12, learning_rate=0.04)

# evaluate on test data
test_loss = evaluate(x_test[:100], y_test[:100])
print("Test loss:", test_loss)
