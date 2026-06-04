import numpy as np
import datasets

# TODO: plot loss, fix bugs, test accuracy

# d_i is the # of input dimensions, d_o is the # of output dimensions
# d_h is the # of hidden dimensions, K is the number of layers (including the input layer)
# B is the batch_size
def init_network(d_i, d_o, d_h, K):
    layers = [np.zeros((d_h, 1))] * (K + 1) # Include the input layer
    bias   = [np.zeros((d_h, 1))] * K
    bias[-1] = np.zeros(( d_o, 1))

    weights = [np.array(0)] * K
    weights[0] = np.zeros((d_h, d_i))
    weights[-1] = np.zeros((d_o, d_h))

    # He intialization
    rng = np.random.default_rng()
    for i in range(1, K - 1):
        variance = 4.0 / (d_i + d_h) if i == 0 else 2.0 / d_h
        weights[i] = rng.normal(0, np.sqrt(variance), size=(d_h, d_h))

    return layers, weights, bias

def cross_entropy_loss(softmax_output, batch_label_y):
    return -(batch_label_y * np.log(softmax_output)).sum()

def backpropagation(y_batch, layers, weights, num_layers):
    dl_dlayers  = [np.array(0)] * num_layers
    dl_dweights = [np.array(0)] * num_layers
    dl_dbias    = [np.array(0)] * num_layers

    # Derivative of the loss with respect to the output layer
    dl_dlayers[-1] = np.transpose(layers[-1] - y_batch)

    # Compute the local gradient, update the global gradient and pass the
    # global gradient to previous layers, from right to left
    for i in range(num_layers - 1, -1, -1):
        dl_dbias[i] = dl_dlayers[i]
        dl_dweights[i] = np.outer(dl_dlayers[i], np.transpose(layers[i]))
        if i > 0:
            relu_deriv = dl_dlayers[i - 1] > 0
            dl_dlayers[i - 1] = relu_deriv * (np.transpose(weights[i]) @ dl_dlayers[i])

    return dl_dweights, dl_dbias

def adam(m, v, t, beta1, beta2, alpha, num_layers, params, gradients):
    for i in range(num_layers):
        m[i] = beta1 * m[i] + (1 - beta1) * gradients[i]
        v[i] = beta2 * v[i] + (1 - beta2) * (gradients[i] ** 2)

        amplified_m = m[i] / (1 - np.pow(beta1, t + 1))
        amplified_v = v[i] / (1 - np.pow(beta2, t + 1))
        params[i] -= alpha * amplified_m / (np.sqrt(amplified_v) + 0.001)

x_train, y_train, x_test, y_test = datasets.to_nparray(*datasets.load_iris())
num_layers, num_batches, num_epochs = 3, 5, 50
batch_size = int(x_train.shape[0] / num_batches)
layers, weights, bias = init_network(4, 3, 10, num_layers)

beta1, beta2 = 0.9, 0.999
initial_learning_rate, gamma, step_size = 0.1, 0.1, 10

m_weights = [np.zeros(weights[i].shape) for i in range(num_layers)]
v_weights = [np.zeros(weights[i].shape) for i in range(num_layers)]
m_bias = [np.zeros(bias[i].shape) for i in range(num_layers)]
v_bias = [np.zeros(bias[i].shape) for i in range(num_layers)]

for epoch in range(num_epochs):
    for _ in range(num_batches):
        # Sample a batch
        rng = np.random.default_rng()
        indices = rng.integers(0, x_train.shape[0], batch_size)

        weight_gradient = [np.zeros_like(weights[i]) for i in range(num_layers)]
        bias_gradient = [np.zeros_like(bias[i]) for i in range(len(bias))]

        for batch_index in indices:
            layers[0] = np.reshape(x_train[batch_index], (x_train.shape[1], 1))
            y = np.reshape(y_train[batch_index], (1, y_train.shape[1]))

            # Forward pass (linear combination + relu)
            for i in range(1, num_layers + 1):
                layers[i] = bias[i - 1] + weights[i - 1] @ layers[i - 1]
                layers[i] = layers[i].clip(0.0)

            # Apply softmax to output layer
            e = np.exp(np.reshape(layers[-1], (1, y_train.shape[1])))
            softmax_output = e / e.sum(axis=1)
            layers[-1] = softmax_output

            loss = cross_entropy_loss(layers[-1], y)
            print(loss)

            # Accumulate gradients for each layer
            dl_dweights, dl_dbias = backpropagation(y, layers, weights, num_layers)
            for layer in range(len(dl_dweights)):
                weight_gradient[layer] += dl_dweights[layer]
                bias_gradient[layer] += dl_dbias[layer]

        alpha = initial_learning_rate * np.pow(gamma, int(epoch / step_size))

        # Adam optimization
        avg_wgradient = [x / batch_size for x in weight_gradient]
        avg_bgradient = [x / batch_size for x in bias_gradient]
        adam(m_weights, v_weights, epoch, beta1, beta2, alpha, num_layers, weights, avg_wgradient)
        adam(m_bias, v_bias, epoch, beta1, beta2, alpha, num_layers, bias, avg_bgradient)
