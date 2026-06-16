# Goal: Implement a CNN following the AlexNet architecture from scratch using nothing but numpy.
#       Vectorize batch operations.
from PIL import Image
from pathlib import Path
import numpy as np

# TODO: run this in a seperate thread?
def load_dataset(path):
  label_classes = {"angry": 0, "confused": 1, "disgust": 2, "fear": 3,
                   "happy": 4, "neutral": 5, "sad": 6, "shy": 7, "surprise": 8}
  labels = np.zeros((16192, len(label_classes)))
  images = []
  for item in Path(path).iterdir():
    if item.is_dir():
      for i, file in enumerate(item.iterdir()):
        one_hot_index = label_classes[file.parent.name]
        labels[i][one_hot_index] = 1

        file = Image.open(file); file.load()
        image = np.array(file)
        print(image.shape)
        file.close()
        images.append(image)
  return images, labels

images, labels = load_dataset("../data/fane/fane_data")
print(labels.shape, len(images))