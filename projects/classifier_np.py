import numpy as np
import datasets

# d_i is the # of input dimensions, d_o is the # of output dimensions
# d_h is the # of hidden dimensions, K is the number of layers (including the input layer)
# B is the batch_size
def init_network(d_i, d_o, d_h, K, B):
    layers = [np.zeros((B, d_h, 1))] * (K + 1) # Include the input layer
    bias   = [np.zeros((B, d_h, 1))] * K
    bias[-1] = np.zeros((B, d_o, 1))

    weights = [np.array(0)] * K
    weights[0] = np.zeros((batch_size, d_h, d_i))
    weights[-1] = np.zeros((batch_size, d_o, d_h))

    # He intialization
    rng = np.random.default_rng()
    for i in range(1, K - 1):
        variance = 4.0 / (d_i + d_h) if i == 0 else 2.0 / d_h
        weights[i] = rng.normal(0, np.sqrt(variance), size=(B, d_h, d_h))

    return layers, weights, bias

def cross_entropy_loss(batch_out_y, batch_label_y):
    # Apply softmax: maps values to a range of 0-1, where they all sum to 1
    e = np.exp(batch_out_y)
    softmax_values = e / e.sum(axis=2)[:, :, None]

    likelyhood = np.log(np.exp(softmax_values).sum(axis=2))
    return -((batch_label_y - likelyhood[:, :, None]).sum())

def backpropagation(loss, layers, weights, bias):
    dloss_dweights = []
    dloss_dbias = []
    dloss_dlayers = []
    return []

x_train, y_train, x_test, y_test = datasets.to_nparray(*datasets.load_iris())
num_layers, num_batches = 3, 5
batch_size = int(x_train.shape[0] / num_batches)
layers, weights, bias = init_network(4, 3, 10, num_layers, batch_size)

for _ in range(num_batches):
    # Sample a batch
    rng = np.random.default_rng()
    indices = rng.integers(0, x_train.shape[0], (1, batch_size))

    out_shape = (batch_size, 1, y_train.shape[1])
    layers[0] = np.reshape(x_train[indices], (batch_size, x_train.shape[1], 1))
    y_batch = np.reshape(y_train[indices], out_shape)

    # Forward pass (linear combination + relu)
    for i in range(1, num_layers + 1):
        layers[i] = bias[i - 1] + weights[i - 1] @ layers[i - 1]
        layers[i] = layers[i].clip(0.0)

    loss = cross_entropy_loss(np.reshape(layers[-1], out_shape), y_batch)
    gradients = backpropagation(loss, layers, weights, bias)

    # TODO: Adam and a learning rate scheduler