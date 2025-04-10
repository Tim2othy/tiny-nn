import numpy as np

from load_mnist import x_test, x_train, y_test, y_train


def p(yt, yp): return 2 * (yp - yt) / yt.size
def s(x):
    h = np.exp(x - np.max(x))
    return h / np.sum(h)
def b(b, w, input, out_err):
    n = np.dot(out_err, w.T)
    w -= 0.04 * np.dot(input.T, out_err)
    b -= 0.04 * np.sum(out_err, axis=0, keepdims=True)
    return n
def f(a,b):return np.random.rand(a,b) - 0.5
w1,b1,w2,b2,w3,b3 = (f(784, 100), f(1, 100), f(100, 50), f(1, 50), f(50, 10), f(1, 10))
t = 60000
e = 0
for i in range(t):
    pixels = x_train[i]
    o1 = np.dot(pixels, w1) + b1
    o2 = np.dot(o1, w2) + b2
    o3 = np.dot(o2, w3) + b3
    r = s(o3)
    b(b1, w1, pixels, b(b2, w2, o1, b(b3, w3, o2, p(y_train[i], r))))
    e += np.mean(np.power(y_train[i] - r, 2))
    if (i + 1) % 7500 == 0:
        print(f"At {i + 1}/{t} the error is {e*8/t:.3f}")
        e = 0
for i in range(10000):
    e += np.mean(np.power(y_test[i] - s(np.dot(np.dot(np.dot(x_test[i], w1) + b1, w2) + b2, w3) + b3), 2))
print(f"Test loss: {e / 10000:.3f}")
