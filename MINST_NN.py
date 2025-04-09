import numpy as np

from activation_Layer import ActivationLayer
from layers import FCLayer, tanh, tanh_prime
from losses import mse, mse_prime
from mnist_loader import MnistDataloader, paths, to_categorical
from network import Network

mnist_dataloader = MnistDataloader(
    paths["train_img"],
    paths["train_lab"],
    paths["test_img"],
    paths["test_lab"],
)

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()


# Convert to numpy arrays first
x_train = np.array(x_train, dtype=np.float32)
x_test = np.array(x_test, dtype=np.float32)


# Preprocess the training data
# Reshape to (num_samples, 1, 28*28) and normalize to range [0, 1]
x_train = x_train.reshape(x_train.shape[0], 1, 28 * 28).astype("float32") / 255
# Convert labels to one-hot encoding
y_train = to_categorical(y_train)

# Preprocess the test data
x_test = x_test.reshape(x_test.shape[0], 1, 28 * 28).astype("float32") / 255
y_test = to_categorical(y_test)

# Create the network
net = Network(mse, mse_prime)
net.add(FCLayer(28 * 28, 100))  # input_shape=(1, 28*28)    ;   output_shape=(1, 100)
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(100, 50))  # input_shape=(1, 100)      ;   output_shape=(1, 50)
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(50, 10))  # input_shape=(1, 50)       ;   output_shape=(1, 10)
net.add(ActivationLayer(tanh, tanh_prime))

# train the network
net.fit(x_train[:1000], y_train[:1000], epochs=12, learning_rate=0.04)

# evaluate on test data
test_loss = net.evaluate(x_test[:100], y_test[:100])
print("Test loss:", test_loss)

# visualize the network on a few samples
out = net.predict(x_test[:8])
print("\nPredicted values:")
print(np.round(out, 1))
print("True values:")
print(y_test[:8])
