# Small ResNet model implemented from scratch.
import numpy as np
import matplotlib.pyplot as plt
import datasets

def relu(data):
    pass

def softmax(data):
    pass

def convolution(data, kernel, stride, padding):
    pass

def convolutional(data, kernel, bias, stride=1, padding=1):
    pass

def linear(data, weight, bias):
    pass

def batchnorm(data, gamma, delta):
    pass

def residual(data, prev_block_out, projection):
    pass

def residual_block(in_data, weights, biases, widx, bidx, downsample):
    l1 = batchnorm(in_data, weights[widx], biases[bidx])
    l2 = relu(l1)
    l3 = convolutional(l2, weights[widx+1], biases[bidx+1], stride=2 if downsample else 1)

    l4 = batchnorm(l3, weights[widx+2], biases[bidx+2])
    l5 = relu(l4)
    l6 = convolutional(l5, weights[widx+3], biases[bidx+3])

    l7 = residual(l6, in_data, weights[widx+4] if downsample else None)

    layers = [l1, l2, l3, l4, l5, l6, l7]
    new_bidx, new_widx = bidx + 4, widx + 5 if downsample else widx + 4
    return layers, new_widx, new_bidx


def forward(in_data, weights, biases):
    in_layer = convolutional(in_data, weights[0], biases[0])

    layers, widx, bidx = [in_layer], 1, 1
    for block in range(6):
        downsample = block == 2 or block == 4
        block_layers, widx, bidx = \
            residual_block(layers[-1], weights, biases, widx, bidx, downsample)
        layers.extend(block_layers)

    assert widx == len(weights) - 1 and bidx == len(biases) - 1
    out_layer = linear(layers[-1], weights[-1], biases[-1])
    layers.append(softmax(out_layer))

batches, classes = datasets.load_cifar10("../data/cifar10")
imgs, labels = batches[0]

layer_weight_shapes = [
    # First convolutional layer kernel
    (16, 3, 3, 3),
    # First two resdiual blocks (batchnorm gammas, kernels)
    (16, 1), (16, 16, 3, 3), (16, 1), (16, 16, 3, 3),
    (16, 1), (16, 16, 3, 3), (16, 1), (16, 16, 3, 3),
    # Second two resdiual blocks (batchnorm gammas, kernels, projections)
    (16, 1), (32, 16, 3, 3), (32, 1), (32, 32, 3, 3), (32, 16, 1, 1),
    (32, 1), (32, 32, 3, 3), (32, 1), (32, 32, 3, 3),
    # Third two resdiual blocks (batchnorm gammas, kernels, projections)
    (32, 1), (64, 32, 3, 3), (64, 1), (64, 64, 3, 3), (64, 32, 1, 1),
    (64, 1), (64, 64, 3, 3), (64, 1), (64, 64, 3, 3),
    # Last linear layer weight
    (10, 64)
]

layer_bias_shapes = [
    # First convolutional layer bias
    (16, 1),
    # First two resdiual blocks (batchnorm delta, biases)
    (16, 1), (16, 1), (16, 1), (16, 1),
    (16, 1), (16, 1), (16, 1), (16, 1),
    # Second two resdiual blocks (batchnorm delta, biases)
    (16, 1), (32, 1), (32, 1), (32, 1),
    (16, 1), (32, 1), (32, 1), (32, 1),
    # Third two resdiual blocks (batchnorm delta, biases)
    (32, 1), (64, 1), (64, 1), (64, 1),
    (32, 1), (64, 1), (64, 1), (64, 1),
    # Last convolutional layer bias
    (10, 1)
]

weights = [np.zeros(shape) for shape in layer_weight_shapes]
biases = [np.zeros(shape) for shape in layer_bias_shapes]
forward(imgs[0], weights, biases)