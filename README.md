# tiny-inn
[`small_net.py`](small_net.py) is a very compact, minimal neural net trained on the MNIST dataset.

[`tiny_net.py`](tiny_net.py) is an even smaller neural net, fitting into just 72 lines, while still being neatly formatted. It does this by not using classes and removing almost everything that isn't absolutely necessary. The network only consists of 3 fully connected layers and one softmax layer. It achieves a test accuracy of 98.8%.

I mostly created this repo to understand how neural networks are implemented in python. And I wanted to see how small I could get a network without impacting performance.

I might add a tiny transformer next.
