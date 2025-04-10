import numpy as np

from load_mnist import x_test, x_train, y_test, y_train


def p(yt, yp): return 2 * (yp - yt) / yt.size
def s(x): return np.exp(x - np.max(x, axis=1, keepdims=True)) / np.sum(np.exp(x - np.max(x, axis=1, keepdims=True)), axis=1, keepdims=True)
def b(b, w, input, out_err):
    n = np.dot(out_err, w.T)
    w -= 0.04 * np.dot(input.T, out_err)
    b -= 0.04 * np.sum(out_err, axis=0, keepdims=True)
    return n


w1 = np.random.rand(28 * 28, 100) - 0.5
b1 = np.random.rand(1, 100) - 0.5
w2 = np.random.rand(100, 50) - 0.5
b2 = np.random.rand(1, 50) - 0.5
w3 = np.random.rand(50, 10) - 0.5
b3 = np.random.rand(1, 10) - 0.5

e = 0
for i in range(60000):
    pixels = x_train[i]
    o1 = np.dot(pixels, w1) + b1
    o2 = np.dot(o1, w2) + b2
    o3 = np.dot(o2, w3) + b3
    r = s(o3)
    b(b1, w1, pixels, b(b2, w2, o1, b(b3, w3, o2, p(y_train[i], r))))
    e += np.mean(np.power(y_train[i] - r, 2))
    if (i + 1) % 7500 == 0:
        print(f"At {i + 1}/{60000} the error is {e / 7500:.3f}")
        e = 0
for i in range(10000):
        e += np.mean(np.power(y_test[i] - s(np.dot(np.dot(np.dot(x_test[i], w1) + b1, w2) + b2, w3) + b3), 2))
print(f"Test loss: {e / 10000:.3f}")
