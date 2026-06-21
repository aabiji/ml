import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view

np.set_printoptions(precision=4)

def load_dataset(path, classes, img_dim, batch_size):
  min_file_count = 10000000
  for folder in  Path(path).iterdir():
    file_count = sum(1 for item in folder.iterdir() if item.is_file())
    min_file_count = min(min_file_count, file_count)
  imgs_per_class = min(int(batch_size / len(classes)), min_file_count)

  imgs = np.zeros((batch_size, 3, img_dim[0], img_dim[1]))
  labels = np.zeros((batch_size, len(classes), 1))

  for i, folder in enumerate(Path(path).iterdir()):
    for j, item in enumerate(folder.iterdir()):
      if j == imgs_per_class:
        break

      count = i * imgs_per_class + j
      one_hot = np.zeros((len(classes), 1))
      one_hot[classes[folder.name]] = 1
      labels[count] = one_hot

      file = Image.open(item)
      resized = ImageOps.pad(file, img_dim, color="black",
        centering=(0, 0), method=Image.Resampling.NEAREST)
      arr = np.array(resized).astype(np.float32) / 255.0
      imgs[count] = arr.T
      file.close()

  return imgs, labels

def expand(data, pad, dilation):
  output = data

  # Dilate - Insert a number of zeroes between elements in the array in the specific axes
  if dilation > 1:
    new_shape = list(data.shape)
    slices = [slice(None) for _ in range(data.ndim)]
    last = len(data.shape) - 1

    # Update the sizes of the last two axes
    new_shape[last]= data.shape[last] + (data.shape[last] - 1) * (dilation - 1)
    new_shape[last - 1] = data.shape[last - 1] + (data.shape[last - 1] - 1) * (dilation - 1)

    slices[last] = slice(None, None, dilation)
    slices[last - 1] = slice(None, None, dilation)

    output = np.zeros(new_shape, dtype=data.dtype)
    output[tuple(slices)] = data

  # Pad - Only pad the last two dimensions
  if pad > 0:
    dims = [(0, 0) for _ in range(len(data.shape))]
    dims[-1] = dims[-2] = (pad, pad)
    output = np.pad(output, pad_width=tuple(dims), mode="constant", constant_values=0)

  return output

# layer_shape   = (num_channels, height, width)
# weights_shape = (output channels, input channels, size, size)
def cnn_layer_shape(layer_shape, kernel_shape, stride, pad):
  return (
    kernel_shape[0],
    int((layer_shape[1] + 2 * pad - kernel_shape[2]) / stride) + 1,
    int((layer_shape[2] + 2 * pad - kernel_shape[2]) / stride) + 1
  )

def convolution(layer, kernel, stride, layer_pad, layer_dilation, kernel_pad, kernel_dilation):
  # layer shape  = (input channels, height, width)
  layer = expand(layer, layer_pad, layer_dilation)

  #   foward pass: (output channels, input channels, kernel size, kernel size)
  # backward pass: (output channels, kernel size, kernel size)
  kernel = expand(kernel, kernel_pad, kernel_dilation)

  axes = (layer.ndim - 2, layer.ndim - 1)
  kernel_shape = (kernel.shape[-2], kernel.shape[-1])

  # (input channels, new height, new width, kernel_size, kernel_size)
  regions = sliding_window_view(layer, kernel_shape, axes)[:, ::stride, ::stride, :, :]

  # weight gradient pass: (output channels, input channels, height, width)
  if kernel.ndim == 3:
    return np.einsum("ihwyx,oyx->oihw", regions, kernel)

  # layer gradient pass: (input channels, height, width)
  if layer_pad == kernel.shape[-1] - 1:
    return np.einsum("ohwyx,ioyx->ihw", regions, kernel)

  # standard convolution: (output channels, height, width)
  return np.einsum("ihwyx,oiyx->ohw", regions, kernel)

def initialize_params(shape):
  out_size, in_size = 1, 1
  if len(shape) == 4:
    in_size = shape[1] * shape[2] * shape[3]
    out_size = shape[0]
  else:
    out_size, in_size = shape[0], shape[1]

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

def cross_entropy_loss(y, true_y):
  return y - true_y

class Linear:
  def __init__(self, weights_shape):
    self.weights, self.bias = initialize_params(weights_shape)
    self.data = None
    self.weights_gradient = None
    self.bias_gradient = None

  def forward(self, prev_layer):
    self.data = self.bias + self.weights @ prev_layer

  def backward(self, prev_layer, gradient):
    self.bias_gradient = gradient
    self.weights_gradient = np.outer(gradient, self.data.T)
    return self.weights.T @ gradient

class Convolutional:
  def __init__(self, weights_shape, stride, padding):
    # Note that each channel gets its own bias
    self.kernel, self.biases = initialize_params(weights_shape)
    self.stride = stride
    self.padding = padding
    self.data = None
    self.kernel_gradient = None
    self.bias_gradient = None

  def forward(self, prev_layer):
    output = convolution(prev_layer, self.kernel, self.stride, self.padding, 1, 0, 1)
    self.data = self.biases.squeeze()[:, None, None] + output

  def backward(self, prev_layer, gradient):
    self.bias_gradient = gradient
    self.kernel_gradient = convolution(prev_layer, gradient, 1, self.padding, 1, 0, self.stride)

    # Turn (C_o, C_i, K, K) ->  (C_i, C_o, K, K) and flip the kernel contents
    transposed = np.transpose(self.kernel, (1, 0, 2, 3))
    rotated = np.rot90(transposed, k=2, axes=(2, 3))
    kernel_size = self.kernel.shape[-1]

    gradient = convolution(gradient, rotated, 1, kernel_size - 1, self.stride, 0, 1)
    if self.padding > 0: # Strip padding
      gradient = gradient[:, self.padding:-self.padding, self.padding:-self.padding]
    return gradient

class Maxpool:
  def __init__(self, kernel_size, stride, flatten):
    self.data = None
    self.original_shape = None # Before the maxpool
    self.max_coordinates = []
    self.kernel_size = kernel_size
    self.stride = stride
    self.flatten = flatten

  def forward(self, prev_layer):
    # Max pooling to reduce layer dimensions
    s = (prev_layer.shape[0], prev_layer.shape[0], self.kernel_size, self.kernel_size)
    self.data = np.zeros(cnn_layer_shape(prev_layer.shape, s, self.stride, 0))
    self.original_shape = prev_layer.shape
    channels, height, width = self.data.shape

    for k in range(channels):
      for y in range(0, height):
        for x in range(0, width):
          a, b = y * self.stride, y * self.stride + self.kernel_size
          c, d = x * self.stride, x * self.stride + self.kernel_size
          region = prev_layer[k, a:b, c:d]
          self.data[k, y, x] = np.max(region)

          region_max_idx = np.argmax(region)
          region_y, region_x = np.unravel_index(region_max_idx, region.shape)
          self.max_coordinates.append((k, a + region_y, c + region_x))

    if self.flatten: # Turn into a column vector
      self.data = self.data.flatten()[:, None]

  def backward(self, prev_layer, gradient):
    # The derivative of maxpool is max unpooling of the gradient
    gradient = gradient.flatten()
    new_grad = np.zeros(self.original_shape)

    for i, val in enumerate(gradient):
      k, y, x = self.max_coordinates[i]
      new_grad[k, y, x] = val

    return new_grad

class ReLU:
  def __init__(self):
    self.data = None

  def forward(self, prev_layer):
    self.data = prev_layer.clip(0.0)

  def backward(self, prev_layer, gradient):
    return (self.data > 0) * gradient

classes = {
  "angry": 0, "confused": 1, "disgust": 2, "fear": 3,
  "happy": 4, "neutral": 5, "sad": 6, "shy": 7, "surprise": 8
}
# TODO: need to split into train, test and validation sets
imgs, labels = load_dataset("../data/fane/fane_data/", classes, (227, 227), 9)

layers = [
  Convolutional((96,  3, 11, 11), 4, 0), ReLU(), Maxpool(3, 2, False),
  Convolutional((256, 96, 5,  5), 1, 2), ReLU(), Maxpool(3, 2, False),
  Convolutional((384, 256, 3, 3), 1, 1), ReLU(),
  Convolutional((384, 384, 3, 3), 1, 1), ReLU(),
  Convolutional((256, 384, 3, 3), 1, 1), ReLU(), Maxpool(3, 2, True),
  Linear((4096, 9216)), ReLU(),
  Linear((4096, 4096)), ReLU(),
  Linear((4096, 4096)), ReLU(),
  Linear((9,    4096))
]
sample_x, sample_y = imgs[0], labels[0]

# Forward propagation
for i, layer in enumerate(layers):
  prev_layer = sample_x if i == 0 else layers[i - 1].data
  layer.forward(prev_layer)
y = softmax(layers[-1].data)

# Backward propagation
gradient = cross_entropy_loss(y, sample_y)
for i in range(len(layers) - 1, -1, -1):
  prev_layer = sample_x if i == 0 else layers[i - 1].data
  gradient = layers[i].backward(prev_layer, gradient)

answer = np.zeros(y.shape)
answer[np.argmax(y)] = 1
print("Output", y, answer)
