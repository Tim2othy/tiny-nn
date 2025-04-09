from keras.datasets import mnist
from keras.utils import to_categorical  # Corrected import statement

from activation_Layer import ActivationLayer
from activations import tanh, tanh_prime
from fc_layer import FCLayer
from losses import mse, mse_prime
from network import Network

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the training data
# Reshape to (num_samples, 1, 28*28) and normalize to range [0, 1]
x_train = x_train.reshape(x_train.shape[0], 1, 28 * 28).astype("float32") / 255
# Convert labels to one-hot encoding
y_train = to_categorical(y_train)

# Preprocess the test data
x_test = x_test.reshape(x_test.shape[0], 1, 28 * 28).astype("float32") / 255
y_test = to_categorical(y_test)

# Create the network
net = Network()
net.add(FCLayer(28 * 28, 100))  # input_shape=(1, 28*28)    ;   output_shape=(1, 100)
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(100, 50))  # input_shape=(1, 100)      ;   output_shape=(1, 50)
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(50, 10))  # input_shape=(1, 50)       ;   output_shape=(1, 10)
net.add(ActivationLayer(tanh, tanh_prime))

# Set loss and train the network
net.use(mse, mse_prime)
net.fit(x_train[:20], y_train[:20], epochs=500, learning_rate=0.04)

# Test the network on a few samples
out = net.predict(x_test[:3])
print("\nPredicted values:")
print(out)
print("True values:")
print(y_test[:3])
