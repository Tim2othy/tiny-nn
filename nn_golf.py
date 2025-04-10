import numpy as np

from load_mnist import x_test, x_train, y_test, y_train


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size
def softmax(x):
    return np.exp(x - np.max(x, axis=1, keepdims=True)) / np.sum(np.exp(x - np.max(x, axis=1, keepdims=True)), axis=1, keepdims=True)
def backprop_fc(bias, weights, input, output_error):
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

loss = 0
for i in range(60000):
    pixels = x_train[i]
    output1 = np.dot(pixels, w1) + b1
    output2 = np.dot(output1, w2) + b2
    output3 = np.dot(output2, w3) + b3
    prediction = softmax(output3)
    backprop_fc(b1, w1, pixels, backprop_fc(b2, w2, output1, backprop_fc(b3, w3, output2, mse_prime(y_train[i], prediction))))
    loss += np.mean(np.power(y_train[i] - prediction, 2))
    if (i + 1) % 7500 == 0:
        print(f"At {i + 1}/{60000} the error is {loss / 7500:.3f}")
        loss = 0
for i in range(10000):
        loss += np.mean(np.power(y_test[i] - softmax(np.dot(np.dot(np.dot(x_test[i], w1) + b1, w2) + b2, w3) + b3), 2))
print(f"Test loss: {loss / 10000:.3f}")
