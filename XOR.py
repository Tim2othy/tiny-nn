import numpy as np

from activation_Layer import ActivationLayer
from layers import FCLayer, tanh, tanh_prime
from losses import mse, mse_prime
from network import Network

# training data
x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# network
net = Network(mse, mse_prime)
net.add(FCLayer(2, 3))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(3, 1))
net.add(ActivationLayer(tanh, tanh_prime))


# train
net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)


# test
out = net.predict(x_train)


print(out)
