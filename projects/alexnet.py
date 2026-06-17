# Goal: Implement a CNN following the AlexNet architecture from scratch using nothing but numpy.
#       Vectorize batch operations.
from PIL import Image, ImageOps
from pathlib import Path
import numpy as np

"""
imgs_per_class = min(batch_size / len(classes), min_file_count)
"""

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

# Convolution with zero padding and dilation=1
def convolution(in_layer, kernels, stride, pad):
  pad_width = ((pad, pad), (pad, pad), (0, 0))
  padded = np.pad(in_layer, pad_width=pad_width, mode="constant", constant_values=0)

  # kernels shape = (num kernels, size, size, input layer channels)
  out_channels, kernel_size = kernels.shape[0], kernels.shape[1]
  half = int(kernel_size / 2)
  inset = pad if pad > 0 else half

  # layer shape = (height, width, num_channels)
  out_layer = np.zeros((
    int((in_layer.shape[0] + 2 * pad - kernel_size) / stride) + 1,
    int((in_layer.shape[1] + 2 * pad - kernel_size) / stride) + 1,
    out_channels
  ))

  for k in range(out_channels):
    for y in range(0, out_layer.shape[0], stride):
      for x in range(0, out_layer.shape[1], stride):
        a, b = inset + y - half, inset + y + half + 1
        c, d = inset + x - half, inset + x + half + 1
        out_layer[y, x, k] = np.einsum("ijk,ijk->ij", padded[a:b, c:d], kernels[k]).sum()

  return out_layer

def maxpool(in_layer, kernel_size, stride):
  # layer shape = (height, width, num_channels)
  out_layer = np.zeros((
    int((in_layer.shape[0] - kernel_size) / stride) + 1,
    int((in_layer.shape[1] - kernel_size) / stride) + 1,
    in_layer.shape[2]
  ))

  for k in range(out_layer.shape[2]):
    for y in range(0, out_layer.shape[0]):
      for x in range(0, out_layer.shape[1]):
        a, b = y * stride, y * stride + kernel_size + 1
        c, d = x * stride, x * stride + kernel_size + 1
        out_layer[y, x, k] = np.max(in_layer[a:b, c:d, k])

  return out_layer

classes = {
  "angry": 0, "confused": 1, "disgust": 2, "fear": 3,
  "happy": 4, "neutral": 5, "sad": 6, "shy": 7, "surprise": 8
}
# TODO: need to split into train, test and validation sets
imgs, labels = load_dataset("../data/fane/fane_data/", classes, np.array([227, 227, 3]), 500)

# TODO: max pooling and fully connected layers, proper forward pass
layer_kernel_info = [
  [4, 0, (96, 11, 11, 3)],
  [1, 2, (256, 5, 5, 96)],
  [1, 1, (384, 3, 3, 256)],
  [1, 1, (384, 3, 3, 384)],
  [1, 1, (256, 3, 3, 384)],
]

out_layer = imgs[0]
for i, (stride, padding, size) in enumerate(layer_kernel_info):
  kernels = np.random.random(size=size)
  out_layer = convolution(out_layer, kernels, stride, padding)
  if i == 0 or i == 1 or i == 4:
    out_layer = maxpool(out_layer, 3, 2)
  print(out_layer.shape)
