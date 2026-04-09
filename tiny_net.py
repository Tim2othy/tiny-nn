import numpy as np

from load_mnist import x_test, x_train, y_test, y_train


def mse(y_true, y_pred) -> float:
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred) -> float:
    return 2 * (y_pred - y_true) / y_true.size


def softmax(x) -> np.ndarray:
    exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

fp = lambda input, w, b: np.dot(input, w) + b

def backprop_fc(bias, weights, input, output_error) -> float:
    input_error = np.dot(output_error, weights.T)
    weights -= 0.04 * np.dot(input.T, output_error)
    bias -= 0.04 * np.sum(output_error, axis=0, keepdims=True)
    return input_error


def train():
    train_loss = 0
    for i in range(60000):
        """forward propagation"""
        pixels = x_train[i]
        label = y_train[i]

        output1 = fp(pixels, w1, b1)
        output2 = fp(output1, w2, b2)
        output3 = fp(output2, w3, b3)
        prediction = softmax(output3)

        train_loss += mse(label, prediction)

        if (i + 1) % 7500 == 0:
            print(f"At {i + 1}/{60000} the error is {train_loss / 7500:.3f}")
            train_loss = 0

        """backward propagation"""
        error = mse_prime(label, prediction)
        error = backprop_fc(b3, w3, output2, error)
        error = backprop_fc(b2, w2, output1, error)
        backprop_fc(b1, w1, pixels, error)


def test():
    test_loss = 0
    for i in range(10000):
        """forward propagation"""
        pixels = x_test[i]
        label = y_test[i]

        output1 = fp(pixels, w1, b1)
        output2 = fp(output1, w2, b2)
        output3 = fp(output2, w3, b3)
        prediction = softmax(output3)

        test_loss += mse(label, prediction)
    print(f"Test loss: {test_loss / 10000:.3f}")


w1 = np.random.rand(28 * 28, 100) - 0.5
b1 = np.random.rand(1, 100) - 0.5
w2 = np.random.rand(100, 50) - 0.5
b2 = np.random.rand(1, 50) - 0.5
w3 = np.random.rand(50, 10) - 0.5
b3 = np.random.rand(1, 10) - 0.5

train()
test()
