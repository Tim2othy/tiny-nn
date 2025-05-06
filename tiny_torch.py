import torch as to

from load_mnist import to_torch, x_test, x_train, y_test, y_train

x_test, x_train, y_test, y_train = to_torch(x_test, x_train, y_test, y_train)


def mse(y_true, y_pred):
    return to.mean((y_true - y_pred) ** 2)


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / to.numel(y_true)


def softmax(x):
    exp_values = to.exp(x - to.max(x, dim=1, keepdim=True)[0])
    return exp_values / to.sum(exp_values, dim=1, keepdim=True)


def backprop_fc(bias, weights, input, output_error):
    input_error = to.matmul(output_error, weights.T)
    weights -= 0.04 * to.matmul(input.T, output_error)
    bias -= 0.04 * to.sum(output_error, dim=0, keepdim=True)
    return input_error


def train():
    train_loss = 0
    for i in range(60000):
        """forward propagation"""
        pixels = x_train[i]

        output1 = to.matmul(pixels, w1) + b1
        output2 = to.matmul(output1, w2) + b2
        output3 = to.matmul(output2, w3) + b3
        prediction = softmax(output3)

        """backward propagation"""
        error = mse_prime(y_train[i], prediction)
        error = backprop_fc(b3, w3, output2, error)
        error = backprop_fc(b2, w2, output1, error)
        error = backprop_fc(b1, w1, pixels, error)

        train_loss += mse(y_train[i], prediction)
        if (i + 1) % 7500 == 0:
            print(f"At {i + 1}/{60000} the error is {train_loss / 7500:.3f}")
            train_loss = 0


def test():
    test_loss = 0
    for i in range(10000):
        """forward propagation"""
        pixels = x_test[i]

        output = to.matmul(pixels, w1) + b1
        output = to.matmul(output, w2) + b2
        output = to.matmul(output, w3) + b3
        prediction = softmax(output)

        test_loss += mse(y_test[i], prediction)
    print(f"Test loss: {test_loss / 10000:.3f}")


w1 = to.rand(28 * 28, 100) - 0.5
b1 = to.rand(1, 100) - 0.5
w2 = to.rand(100, 50) - 0.5
b2 = to.rand(1, 50) - 0.5
w3 = to.rand(50, 10) - 0.5
b3 = to.rand(1, 10) - 0.5

train()
test()
