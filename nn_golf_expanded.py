from numpy import dot, exp, max, mean, random, sum
from load_mnist import xs, xt, ys, yt

softmax = lambda x: exp(x - max(x)) / sum(exp(x - max(x)))
D = dot

def b(bias, weights, input, output_error):
    g = D(output_error, weights.T)
    weights -= 0.02 * D(input.T, output_error)
    bias -= 0.02 * sum(output_error)
    return g


w1 = random.rand(28 * 28, 100) - 0.5
w2 = random.rand(100, 50) - 0.5
w3 = random.rand(50, 10) - 0.5
b1 = random.rand(1, 100) - 0.5
b2 = random.rand(1, 50) - 0.5
b3 = random.rand(1, 10) - 0.5

error = 0


for i in range(60000):
    output1 = D(xt[i], w1) + b1
    output2 = D(output1, w2) + b2
    output3 = D(output2, w3) + b3
    prediction = softmax(output3)

    error1 = (prediction - yt[i]) / prediction.size
    error2 = b(b3, w3, output2, error1)
    error3 = b(b2, w2, output1, error2)
    b(b1, w1, xt[i], error3)

    error += mean((yt[i] - prediction) ** 2)

    if (i + 1) % 10000 == 0:
        print(f"At {i+1}/{60000} the error is {error/10000}")
        error = 0

for i in range(10000):
    error += mean((ys[i] - softmax(D(D(D(xs[i], w1) + b1, w2) + b2, w3) + b3)) ** 2)

print(f"Test loss: {error/10000}")
