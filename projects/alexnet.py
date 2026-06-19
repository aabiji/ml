from PIL import Image, ImageOps
from pathlib import Path
import numpy as np

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
  # Pad - Add surrounding zeroes
  pad_width = ((0, 0), (pad, pad), (pad, pad))
  data = np.pad(data, pad_width=pad_width, mode="constant", constant_values=0)
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

  return output


# layer_shape   = (num_channels, height, width)
# weights_shape = (output channels, input channels, size, size)
def cnn_layer_shape(layer_shape, kernel_shape, stride, pad):
  return (
    kernel_shape[0],
    int((layer_shape[1] + 2 * pad - kernel_shape[2]) / stride) + 1,
    int((layer_shape[2] + 2 * pad - kernel_shape[2]) / stride) + 1
  )


def convolution(in_layer, kernel, stride, pad, dilation, input_expanded=True):
  out_layer = np.zeros(cnn_layer_shape(in_layer.shape, kernel.shape, stride, pad))
  out_channels, out_height, out_width = out_layer.shape

  kernel_size = kernel.shape[2]
  half = int(kernel_size / 2)
  inset = pad if pad > 0 else half

  if input_expanded:
    in_layer = expand(in_layer, pad, dilation)
  else:
    kernel = expand(kernel, pad, dilation)

  for k in range(out_channels):
    for y in range(0, out_height, stride):
      for x in range(0, out_width, stride):
        a, b = inset + y - half, inset + y + half + 1
        c, d = inset + x - half, inset + x + half + 1
        #print(in_layer[:, a:b, c:d].shape, kernel[k].shape, in_layer.shape)
        out_layer[k, y, x] = np.einsum("ijk,ijk->jk", in_layer[:, a:b, c:d], kernel[k]).sum()

  return out_layer


def maxpool(in_layer, kernel_size, stride):
  kernel_shape = (in_layer.shape[0], in_layer.shape[0], kernel_size, kernel_size)
  out_layer = np.zeros(cnn_layer_shape(in_layer.shape, kernel_shape, stride, 0))
  out_channels, out_height, out_width = out_layer.shape

  for k in range(out_channels):
    for y in range(0, out_height):
      for x in range(0, out_width):
        a, b = y * stride, y * stride + kernel_size + 1
        c, d = x * stride, x * stride + kernel_size + 1
        out_layer[k, y, x] = np.max(in_layer[k, a:b, c:d])

  return out_layer


def initialize_params(shape):
  out_size, *_, in_size = shape
  variance = 2.0 / in_size
  if in_size != out_size:
    variance = 4.0 / (in_size + out_size)

  rng = np.random.default_rng()
  weight = rng.normal(0.0, np.sqrt(variance), shape)
  bias = np.zeros((out_size, 1))
  return weight, bias


class Linear:
  def __init__(self, weights_shape):
    self.weights, self.bias = initialize_params(weights_shape)
    self.data = None
    self.weights_gradient = None
    self.bias_gradient = None

  def forward(self, prev_layer, should_relu):
    self.data = self.bias + self.weights @ prev_layer
    if should_relu:
      self.data = self.data.clip(0.0)

  def backward(self, gradient, next_layer):
    self.bias_gradient = gradient
    self.weights_gradient = gradient @ self.data.T
    return (next_layer > 0) * (self.weights.T @ gradient)

  def softmax(self):
    e = np.exp(self.data - np.max(self.data))
    self.data = e / np.sum(e)


class Convolutional:
  def __init__(self, weights_shape, stride, padding, maxpool_dims=None):
    # Note that each channel gets its own bias
    self.kernel, self.biases = initialize_params(weights_shape)
    self.stride = stride
    self.padding = padding
    self.maxpool_dims = maxpool_dims
    self.data = None
    self.kernel_gradient = None
    self.bias_gradient = None

  def forward(self, prev_layer):
    output = convolution(prev_layer, self.kernel, self.stride, self.padding, 1)
    self.data = self.biases.squeeze()[:, None, None] + output

    if self.maxpool_dims is not None:
      pooled = maxpool(self.data, 3, 2)
      self.data = np.reshape(pooled, self.maxpool_dims)

    self.data = self.data.clip(0.0)

  def backward(self, gradient):
    print(gradient.shape)

    # TODO: back propagatoin on the maxpool layer + its preceding layer

    # TODO: apply maxpool derivative
    self.bias_gradient = gradient
    self.kernel_gradient = convolution(self.data, gradient, 1, 0, self.stride, False)

    rotated = np.rot90(self.kernel, k=2, axes=(0, 1))
    return convolution(gradient, rotated, 1, self.stride, self.stride)


classes = {
  "angry": 0, "confused": 1, "disgust": 2, "fear": 3,
  "happy": 4, "neutral": 5, "sad": 6, "shy": 7, "surprise": 8
}
# TODO: need to split into train, test and validation sets
imgs, labels = load_dataset("../data/fane/fane_data/", classes, (227, 227), 9)

layers = [
  Convolutional((96,  3, 11, 11), 4, 0, maxpool_dims=(96,  27, 27)),
  Convolutional((256, 96, 5,  5), 1, 2, maxpool_dims=(256, 13, 13)),
  Convolutional((384, 256, 3, 3), 1, 1),
  Convolutional((384, 384, 3, 3), 1, 1),
  Convolutional((256, 384, 3, 3), 1, 1, maxpool_dims=(9216, 1)),
  Linear((4096, 9216)),
  Linear((4096, 4096)),
  Linear((4096, 4096)),
  Linear((9,    4096))
]
sample_x, sample_y = imgs[0], labels[0]

# Forward propagation
for i, layer in enumerate(layers):
  if type(layer) is Linear:
    layer.forward(layers[i - 1].data, i != len(layers) - 1)
  else:
    layer.forward(sample_x if i == 0 else layers[i - 1].data)
layers[-1].softmax()

# Backward propagation
gradient = layers[-1].data - sample_y # Derivative of cross entropy loss
for i in range(len(layers) - 1, 0, -1):
  if type(layers[i]) is Linear:
    gradient = layers[i].backward(gradient, layers[i - 1].data)
  else:
    gradient = layers[i].backward(gradient)
