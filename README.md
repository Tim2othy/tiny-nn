# tiny-nn
[`tiny_net.py`](tiny_net.py) is a very compact, minimal neural net trained on the MNIST dataset, fitting into just 72 lines, while still being neatly formatted. It does this by not using classes and removing almost everything that isn't absolutely necessary. The network only consists of 3 fully connected layers and one softmax layer. It achieves a test accuracy of 97.8%.

[`tiny_torch.py`](tiny_torch.py) is exactly the same network except it uses the pytorch instead of the numpy library.

[`nn_golf.py`](nn_golf.py) is the same network, except I tried to make the code as short as possible, not caring about formatting and legibility.

I mostly created this repo to understand how neural networks are implemented in python. And I wanted to see how small I could get a network without impacting performance.

I might add a tiny transformer next.