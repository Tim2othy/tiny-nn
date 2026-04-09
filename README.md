# tiny-nn
[`tiny_nn.py`](tiny_nn.py) is a very compact, minimal neural net trained on the MNIST dataset. I removed everything that isn't absolutely necessary, no classes, simple functions.
The network only consists of 3 fully connected layers, with relu functions in between, and one softmax layer. It achieves a test accuracy of 92.2%.
It's ideal for getting a deep understading of the fundamentals of neural networks without getting distracted by the implementation.

[`tiny_nn_torch.py`](tiny_nn_torch.py) is exactly the same network except it uses the pytorch instead of the numpy library.

[`nn_golf.py`](nn_golf.py) is the same network, except I tried to make the code as short as possible, not caring about formatting and legibility.

I mostly created this repo to understand how neural networks are implemented in python. And I wanted to see how small I could get a network without impacting performance.

I'm planning on adding a tiny transformer next.
