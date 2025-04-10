import numpy as np

from load_mnist import test_data, training_data


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size


def softmax(x):
    exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


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


train_loss = 0
for i in range(60000):
    """forward propagation"""
    pixels = training_data[0][i]

    output1 = np.dot(pixels, w1) + b1
    output2 = np.dot(output1, w2) + b2
    output3 = np.dot(output2, w3) + b3
    prediction = softmax(output3)

    """backward propagation"""
    error = mse_prime(training_data[1][i], prediction)
    error = fc_bp(b3, w3, output2, error)
    error = fc_bp(b2, w2, output1, error)
    error = fc_bp(b1, w1, pixels, error)

    train_loss += mse(training_data[1][i], prediction)
    if (i + 1) % 12000 == 0:
        print(f"At {i + 1}/{60000}   the error is {train_loss / 12000:.3f}")
        train_loss = 0


test_loss = 0
for i in range(10000):
    """forward propagation"""
    pixels = test_data[0][i]

    output = np.dot(pixels, w1) + b1
    output = np.dot(output, w2) + b2
    output = np.dot(output, w3) + b3
    prediction = softmax(output)

    test_loss += mse(test_data[1][i], prediction)

print(f"Test loss: {test_loss / 10000:.3f}")
