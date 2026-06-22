import numpy as np
import os
import random
from PIL import Image, ImageOps
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view

# Implement the following novelties:
# - dropout
# - the same ewight and bias initalized as used in the paper
# - normal response normalization
# - should weight decay be applie
# - data augmentation
# - Visualize loss through each epoch
# Then use cupy when training on Kaggle. Save the weights + biases for each layer to a file for download later.

np.set_printoptions(precision=4)

def select_randomized_files(outer_folder_path, num_files):
  class_folders = Path(outer_folder_path).iterdir()
  all_files = []

  for folder in class_folders:
    with os.scandir(folder.resolve()) as entries:
      all_files.extend([entry.path for entry in entries if entry.is_file()])

  selection_count = min(len(all_files), num_files)
  return random.sample(all_files, selection_count)

def load_dataset(outer_folder_path, classes, img_dim, num_batches, batch_size):
  paths = select_randomized_files(outer_folder_path, num_batches * batch_size)

  imgs = np.zeros((num_batches, batch_size, 3, img_dim[0], img_dim[1]))
  labels = np.zeros((num_batches, batch_size, len(classes), 1))

  for i in range(num_batches):
    for j in range(batch_size):
      path = paths[i * batch_size + j]

      name = Path(path).parent.name
      one_hot = np.zeros((len(classes), 1))
      one_hot[classes[name]] = 1
      labels[i, j] = one_hot

      file = Image.open(path)
      resized = ImageOps.pad(file, img_dim, color="black",
        centering=(0, 0), method=Image.Resampling.NEAREST)
      arr = np.array(resized).astype(np.float32) / 255.0
      imgs[i, j] = arr.T
      file.close()

  return imgs, labels

def expand(data, pad, dilation):
  output = data

  # Dilate - Insert a number of zeroes between
  # elements in the array in the specific axes
  if dilation > 1:
    new_shape = list(data.shape)
    slices = [slice(None) for _ in range(data.ndim)]
    last = len(data.shape) - 1

    # Update the sizes of the last two axes
    new_shape[last] = \
      data.shape[last] + (data.shape[last] - 1) * (dilation - 1)
    new_shape[last - 1] = \
      data.shape[last - 1] + (data.shape[last - 1] - 1) * (dilation - 1)

    slices[last] = slice(None, None, dilation)
    slices[last - 1] = slice(None, None, dilation)

    output = np.zeros(new_shape, dtype=data.dtype)
    output[tuple(slices)] = data

  # Pad - Only pad the last two dimensions
  if pad > 0:
    dims = [(0, 0) for _ in range(len(data.shape))]
    dims[-1] = dims[-2] = (pad, pad)
    output = np.pad(output, pad_width=tuple(dims),
                    mode="constant", constant_values=0)

  return output

# Dimensions:
# B = batch size, H = height, W = width, C = number of channels
# C_i = number of input channels (convolutional) or previous layer width (linear)
# C_o = number of output channels (convolutional) or next layer width (linear)
# K = kernel size
#
# layer_shape = (B, C, H, W), weights_shape = (B, C_o, C_i, K, K)
def cnn_layer_shape(layer_shape, kernel_shape, stride, pad):
  return (
    kernel_shape[1],
    int((layer_shape[-2] + 2 * pad - kernel_shape[-1]) / stride) + 1,
    int((layer_shape[-1] + 2 * pad - kernel_shape[-1]) / stride) + 1
  )

def convolution(layer, kernel, stride, layer_pad,
                layer_dilation, kernel_pad, kernel_dilation):
  layer = expand(layer, layer_pad, layer_dilation) # (B, C_i, H, W)

  # foward pass: (B, C_o, C_i, K, K), backward pass: (B, C_o, K, K)
  kernel = expand(kernel, kernel_pad, kernel_dilation)

  axes = (layer.ndim - 2, layer.ndim - 1)
  kernel_shape = (kernel.shape[-2], kernel.shape[-1])

  # (B, C_i, H, W, K, K)
  regions = \
    sliding_window_view(layer, kernel_shape, axes)[:, :, ::stride, ::stride, :, :]

  # weight gradient: (B, C_o, C_i, H, W)
  if kernel.ndim == 4:
    return np.einsum("bihwyx,boyx->boihw", regions, kernel)

  # layer gradient: (B, C_i, C_o, H, W)
  if layer_pad == kernel.shape[-1] - 1:
    return np.einsum("bohwyx,bioyx->bihw", regions, kernel)

  # standard convolution: (B, C_o, H, W)
  return np.einsum("bihwyx,boiyx->bohw", regions, kernel)

def initialize_params(shape):
  out_size, in_size = 1, 1
  if len(shape) == 5: # weights
    in_size = shape[2] * shape[3] * shape[4]
    out_size = shape[1]
  else: # layers
    out_size, in_size = shape[1], shape[2]

  variance = 2.0 / in_size
  if in_size != out_size:
    variance = 4.0 / (in_size + out_size)

  rng = np.random.default_rng()
  weight = rng.normal(0.0, np.sqrt(variance), shape)
  bias = np.zeros((out_size, 1))
  return weight, bias

def softmax(data):
  e = np.exp(data - np.max(data))
  return e / np.sum(e)

def cross_entropy_loss_gradient(y, true_y):
  return y - true_y

# Flatten data into a column vector while preserving batch size
def flatten(data):
  return data.reshape(data.shape[0], -1, 1)

class Linear:
  def __init__(self, weights_shape):
    self.weights, self.biases = initialize_params(weights_shape)
    self.data = None
    self.weights_gradient = None
    self.biases_gradient = None

  def forward(self, prev_layer):
    self.data = self.biases + self.weights @ prev_layer

  def backward(self, prev_layer, gradient):
    self.biases_gradient = np.sum(gradient, axis=0)
    self.weights_gradient = gradient @ np.swapaxes(prev_layer, -1, -2)
    return np.swapaxes(self.weights, -1, -2) @ gradient

class Convolutional:
  def __init__(self, weights_shape, stride, padding):
    # Note that each channel gets its own bias
    self.weights, self.biases = initialize_params(weights_shape)
    self.stride = stride
    self.padding = padding
    self.data = None
    self.weights_gradient = None
    self.biases_gradient = None

  def forward(self, prev_layer):
    output = convolution(prev_layer, self.weights,
                         self.stride, self.padding, 1, 0, 1)
    self.data = self.biases.squeeze()[:, None, None] + output

  def backward(self, prev_layer, gradient):
    self.biases_gradient = flatten(gradient.sum(axis=(2, 3)))
    self.weights_gradient = convolution(prev_layer, gradient, 1,
                                        self.padding, 1, 0, self.stride)

    # Turn (B, C_o, C_i, K, K) ->  (B, C_i, C_o, K, K) and flip the kernel contents
    transposed = np.transpose(self.weights, (0, 2, 1, 3, 4))
    rotated = np.rot90(transposed, k=2, axes=(3, 4))
    kernel_size = self.weights.shape[-1]

    gradient = convolution(gradient, rotated, 1, kernel_size - 1, self.stride, 0, 1)
    if self.padding > 0: # Strip padding
      gradient = gradient[:, :, self.padding:-self.padding, self.padding:-self.padding]
    return gradient

class Maxpool:
  def __init__(self, kernel_size, stride, flatten):
    self.data = None
    self.argmax = None
    self.old_shape = None
    self.new_shape = None
    self.N = kernel_size
    self.S = stride
    self.flatten = flatten

  def forward(self, prev_layer):
    axes = (prev_layer.ndim - 2, prev_layer.ndim - 1)

    # Extract max value from each (N, N) regions, shape: (B, C, H, W, K, K)
    regions = sliding_window_view(prev_layer, (self.N, self.N), axes)[:, :, ::self.S, ::self.S]
    self.data = np.max(regions, axis=(-2, -1))
    self.old_shape = prev_layer.shape
    self.new_shape = self.data.shape

    # Each value in (B, C, H, W) stores an index from 0 to (N * N - 1)
    b, c, h, w, _, _ = regions.shape
    self.argmax = np.reshape(regions, (b, c, h, w, self.N * self.N)).argmax(axis=-1)

    if self.flatten: # Turn into a column vector
      self.data = flatten(self.data)

  def backward(self, prev_layer, gradient):
    # The derivative of a max pool is a max unpool
    if self.flatten:
      self.data = self.data.reshape(self.new_shape)

    i = self.argmax // self.N
    j = self.argmax % self.N
    b, c, h, w = np.indices(self.data.shape)

    # Assign the max values to their original positions
    gradient = np.zeros(self.old_shape)
    gradient[b, c, h * self.S + i, w * self.S + j] = self.data
    return gradient

class ReLU:
  def __init__(self):
    self.data = None

  def forward(self, prev_layer):
    self.data = prev_layer.clip(0.0)

  def backward(self, prev_layer, gradient):
    return (self.data > 0) * gradient


classes = {
  "angry": 0, "confused": 1, "disgust": 2,
  "fear":  3, "happy":    4, "neutral": 5,
  "sad":   6, "shy":     7, "surprise": 8
}
epochs, num_batches, B = 1, 5, 5
learning_rate = 0.001
num_layers = 9 # Excluding relu and maxpool layers

imgs, labels = load_dataset(
  "../data/fane/fane_data/", classes, (227, 227), num_batches + 1, B)
train_imgs, train_labels = imgs[1:], labels[1:]
test_imgs, test_labels = imgs[0], labels[0]

# (batch size, channels out, channels in, kernel size, kernel size)
weight_shapes = [
  (B,  96, 3, 11, 11), (B,  256, 96, 5, 5), (B, 384, 256, 3, 3),
  (B, 384, 384, 3, 3), (B, 256, 384, 3, 3), (B, 4096, 9216),
  (B, 4096, 4096), (B, 4096, 4096), (B, 9, 4096),
]

layers = [
  Convolutional(weight_shapes[0], 4, 0), ReLU(), Maxpool(3, 2, False),
  Convolutional(weight_shapes[1], 1, 2), ReLU(), Maxpool(3, 2, False),
  Convolutional(weight_shapes[2], 1, 1), ReLU(),
  Convolutional(weight_shapes[3], 1, 1), ReLU(),
  Convolutional(weight_shapes[4], 1, 1), ReLU(), Maxpool(3, 2, True),
  Linear(weight_shapes[5]), ReLU(),
  Linear(weight_shapes[6]), ReLU(),
  Linear(weight_shapes[7]), ReLU(),
  Linear(weight_shapes[8])
]

wmomentum = [np.zeros(weight_shapes[i]) for i in range(num_layers)]
bmomentum = [np.zeros((weight_shapes[i][1], 1)) for i in range(num_layers)]

for epoch in range(epochs):
  for b in range(num_batches):
    print(f"Epoch {epoch}, Batch {b + 1}/{num_batches}")

    batch_x, batch_y = train_imgs[b], train_labels[b]
    wgradients, bgradients = [0] * num_layers, [0] * num_layers

    # Forward propagation
    for i, layer in enumerate(layers):
      prev_layer = batch_x if i == 0 else layers[i - 1].data
      layer.forward(prev_layer)

    # Backward propagation
    count = num_layers - 1
    gradient = cross_entropy_loss_gradient(layers[-1].data, batch_y)
    for i in range(len(layers) - 1, -1, -1):
      prev_layer = batch_x if i == 0 else layers[i - 1].data
      gradient = layers[i].backward(prev_layer, gradient)

      if type(layers[i]) is Linear or type(layers[i]) is Convolutional:
        wgradients[count] = layers[i].weights_gradient
        bgradients[count] = layers[i].biases_gradient
        count -= 1

    indices = [0, 3, 6, 8, 10, 13, 15, 17, 19]
    for j in range(num_layers):
      i = indices[j]
      wmomentum[j] = 0.9 * wmomentum[j] - \
                     0.0005 * learning_rate * layers[i].weights - \
                     learning_rate * wgradients[j] / B
      layers[i].weights += wmomentum[j]

      # Momentum and weight dampening is applied to biases, right?? Should we add weight decay?
      bmomentum[j] = 0.9 * bmomentum[j] - learning_rate * bgradients[j] / B
      layers[i].biases += bmomentum[j]

