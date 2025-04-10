import numpy as np

from load_mnist import test_data, training_data

LR = 0.04


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

        if (i + 1) % round(samples / 8) == 0:
            loss_print /= samples / 8

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


# train the network
train(training_data)

# evaluate on test data
evaluate(test_data)
