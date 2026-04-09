import torch as to

from load_mnist import to_torch, x_test, x_train, y_test, y_train

x_test, x_train, y_test, y_train = to_torch(x_test, x_train, y_test, y_train)


def cross_entropy(y_true, y_pred):
    return -to.log(y_pred[0, to.argmax(y_true)])


def test_error_rate(y_true, y_pred):
    return to.argmax(y_true) != to.argmax(y_pred)

relu = lambda x: to.clamp(x, min=0)
relu_prime = lambda x: (x > 0)


def softmax(x):
    exp_values = to.exp(x - to.max(x, dim=1, keepdim=True)[0])
    return exp_values / to.sum(exp_values, dim=1, keepdim=True)


def backprop_fc(bias: to.Tensor, weights: to.Tensor, input: to.Tensor, output_error: to.Tensor):
    input_error = to.matmul(output_error, weights.T)
    weights -= lr * to.matmul(input.T, output_error)
    bias -= lr * to.sum(output_error, dim=0, keepdim=True)
    return input_error


def train():
    train_loss = 0
    for i in range(60000):
        """forward propagation"""
        pixels = x_train[i]

        output1 = relu(to.matmul(pixels, w1) + b1)
        output2 = relu(to.matmul(output1, w2) + b2)
        output3 = to.matmul(output2, w3) + b3
        prediction = softmax(output3)

        """backward propagation"""
        error = prediction - y_train[i]
        error = backprop_fc(b3, w3, output2, error)
        error *= relu_prime(output2)
        error = backprop_fc(b2, w2, output1, error)
        error *= relu_prime(output1)
        error = backprop_fc(b1, w1, pixels, error)

        train_loss += cross_entropy(y_train[i], prediction)
        if (i + 1) % 7500 == 0:
            print(f"At {i + 1}/{60000} the error is {train_loss / 7500:.3f}")
            train_loss = 0


def test():
    test_loss = 0
    for i in range(10000):
        """forward propagation"""
        pixels = x_test[i]

        output = relu(to.matmul(pixels, w1) + b1)
        output = relu(to.matmul(output, w2) + b2)
        prediction = to.matmul(output, w3) + b3

        test_loss += test_error_rate(y_test[i], prediction)
    print(f"Test loss: {test_loss / 10000:.3f}")

lr = 0.02

w1 = to.rand(28 * 28, 100) - 0.5
b1 = to.rand(1, 100) - 0.5
w2 = to.rand(100, 50) - 0.5
b2 = to.rand(1, 50) - 0.5
w3 = to.rand(50, 10) - 0.5
b3 = to.rand(1, 10) - 0.5

train()
test()
