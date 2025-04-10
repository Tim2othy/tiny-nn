import numpy as np

from load_mnist import test_data, training_data


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size


def softmax(x):
    exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def relu_bp(input, output_error):
    return np.where(input > 0, 1, 0) * output_error


def fc_bp(bias, weights, input, output_error):
    input_error = np.dot(output_error, weights.T)

    weights -= 0.04 * np.dot(input.T, output_error)
    bias -= 0.04 * np.sum(output_error, axis=0, keepdims=True)
    return input_error


w1 = np.random.rand(28 * 28, 100) - 0.5
b1 = np.random.rand(1, 100) - 0.5
w2 = np.random.rand(100, 50) - 0.5
b2 = np.random.rand(1, 50) - 0.5
w3 = np.random.rand(50, 10) - 0.5
b3 = np.random.rand(1, 10) - 0.5


samples = len(training_data[0])
loss_print = 0

for i in range(samples):
    """forward propagation"""
    pixels = training_data[0][i]

    z1 = np.dot(pixels, w1) + b1
    activation1 = np.maximum(0, z1)

    z2 = np.dot(activation1, w2) + b2
    activation2 = np.maximum(0, z2)

    logit = np.dot(activation2, w3) + b3
    prediction = softmax(logit)

    """backward propagation"""
    error = mse_prime(training_data[1][i], prediction)
    error = fc_bp(b3, w3, activation2, error)
    error = relu_bp(z2, error)
    error = fc_bp(b2, w2, activation1, error)
    error = relu_bp(z1, error)
    error = fc_bp(b1, w1, pixels, error)

    loss_print += mse(training_data[1][i], prediction)
    if (i + 1) % round(samples / 8) == 0:
        loss_print /= samples / 8

        print(f"For the sample {i + 1}/{samples}   the error is {loss_print:.4f}")
        loss_print = 0


test_loss = 0

# run network over all samples
for i in range(len(test_data[0])):
    """forward propagation"""
    pixels = test_data[0][i]

    output = np.dot(pixels, w1) + b1
    output = np.maximum(0, output)
    output = np.dot(output, w2) + b2
    output = np.maximum(0, output)
    logit = np.dot(output, w3) + b3
    prediction = softmax(logit)

    test_loss += mse(test_data[1][i], prediction)

print(f"Test loss: {test_loss / len(test_data[0]):.4f}")
