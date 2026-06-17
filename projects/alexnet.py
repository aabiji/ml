# Goal: Implement a CNN following the AlexNet architecture from scratch using nothing but numpy.
#       Vectorize batch operations.
from PIL import Image, ImageOps
from pathlib import Path
import numpy as np

def load_dataset(path, classes, img_dim, batch_size):
  class_folders = Path(path).iterdir()
  min_file_count = 10000000

  for folder in class_folders:
    file_count = sum(1 for item in folder.iterdir() if item.is_file())
    min_file_count = min(min_file_count, file_count)
  imgs_per_class = min(int(batch_size / len(classes)), min_file_count)

  imgs = np.zeros((batch_size, img_dim[0], img_dim[1], img_dim[2]))
  labels = np.zeros((batch_size, len(classes)))

  for i, folder in enumerate(class_folders):
    for j, item in enumerate(folder.iterdir()):
      if j == imgs_per_class:
        break

      file = Image.open(item)
      file.load()

      count = i * imgs_per_class + j
      one_hot = np.zeros(len(classes))
      one_hot[classes[folder.name]] = 1

      resized = ImageOps.pad(file, img_dim, color="black", centering=(0, 0))
      imgs[count] = np.array(resized)
      labels[count] = one_hot

      file.close()

  return imgs, labels

# layer_shape   = (height, width, num channels)
# weights_shape = (output channels, size, size, input channels)
def cnn_layer_shape(layer_shape, weight_shape, stride, pad):
  return (
    int((layer_shape[0] + 2 * pad - weight_shape[1]) / stride) + 1,
    int((layer_shape[1] + 2 * pad - weight_shape[1]) / stride) + 1,
    weight_shape[0]
  )

# Convolution with zero padding and dilation=1
def convolution(in_layer, weights, biases, stride, pad):
  pad_width = ((pad, pad), (pad, pad), (0, 0))
  padded = np.pad(in_layer, pad_width=pad_width, mode="constant", constant_values=0)
  out_layer = np.zeros(cnn_layer_shape(in_layer.shape, weights.shape, stride, pad))

  out_channels, kernel_size = weights.shape[0], weights.shape[1]
  half = int(kernel_size / 2)
  inset = pad if pad > 0 else half

  for k in range(out_channels):
    for y in range(0, out_layer.shape[0], stride):
      for x in range(0, out_layer.shape[1], stride):
        a, b = inset + y - half, inset + y + half + 1
        c, d = inset + x - half, inset + x + half + 1
        out_layer[y, x, k] = np.einsum("ijk,ijk->ij", padded[a:b, c:d], weights[k]).sum()

  out_layer = biases.squeeze()[None, None, :] + out_layer
  return out_layer

def maxpool(in_layer, kernel_size, stride):
  kernel_shape = (in_layer.shape[-1], kernel_size, kernel_size, in_layer.shape[-1])
  out_layer = np.zeros(cnn_layer_shape(in_layer.shape, kernel_shape, stride, 0))

  for k in range(out_layer.shape[2]):
    for y in range(0, out_layer.shape[0]):
      for x in range(0, out_layer.shape[1]):
        a, b = y * stride, y * stride + kernel_size + 1
        c, d = x * stride, x * stride + kernel_size + 1
        out_layer[y, x, k] = np.max(in_layer[a:b, c:d, k])

  return out_layer

def initialize_params(shapes):
  weights = []
  biases = []

  for shape in shapes:
    in_size, out_size = shape[-1], shape[0]
    variance = 2.0 / in_size
    if in_size != out_size:
      variance = 4.0 / (in_size + out_size)

    rng = np.random.default_rng()
    weights.append(rng.normal(0.0, np.sqrt(variance), shape))
    biases.append(np.zeros((out_size, 1)))

  return weights, biases

def forward(input, weights, biases):
  layers = [input]
  strides = [4, 1, 1, 1, 1]
  pads = [0, 2, 1, 1, 1]
  num_layers = len(weights)

  for l in range(num_layers):
    # Fully connected + relu
    if l >= 5:
      layer = biases[l] + weights[l] @ layers[l]
      if l != num_layers - 1:
        layer = layer.clip(0.0)
      layers.append(layer)
      continue

    # Convolutional + relu
    layer = convolution(layers[l], weights[l], biases[l], strides[l], pads[l])
    if l == 0 or l == 1:
      layer = maxpool(layer, 3, 2)
    elif l == 4:
      layer = np.reshape(maxpool(layer, 3, 2), (9216, 1))
    layers.append(layer.clip(0.0))

  # Softmax on the output layer
  e = np.exp(layers[-1])
  layers[-1] = e / np.sum(e)
  return layers

classes = {
  "angry": 0, "confused": 1, "disgust": 2, "fear": 3,
  "happy": 4, "neutral": 5, "sad": 6, "shy": 7, "surprise": 8
}
# TODO: need to split into train, test and validation sets
imgs, labels = load_dataset("../data/fane/fane_data/", classes, np.array([227, 227, 3]), 9)

weights, biases = initialize_params([
  (96, 11, 11, 3), (256, 5,  5, 96), (384, 3, 3, 256), (384, 3, 3, 384), (256, 3, 3, 384),
  (4096, 9216), (4096, 4096), (4096, 4096), (9, 4096),
])
layers = forward(imgs[0], weights, biases)
