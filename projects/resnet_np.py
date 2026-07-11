# Small ResNet model implemented from scratch.
import numpy as np
import matplotlib.pyplot as plt

def load_cifar_dataset(folder):
    import pickle, warnings
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=".*align should be passed as Python or NumPy boolean.*"
    )

    classes = ["airplane", "automobile", "bird", "cat",
               "deer", "dog", "frog", "horse", "ship", "truck"]
    paths = [f"{folder}/data_batch_{i + 1}" for i in range(5)]
    paths.append(f"{folder}/test_batch")
    batches = []

    for path in paths:
        with open(path, "rb") as file:
            dict = pickle.load(file, encoding="bytes")
            data, labels = dict[b"data"], dict[b"labels"]
            images = np.zeros((10000, 3, 32, 32))
            images[:, 0] = data[:, :1024].reshape(10000, 32, 32) / 255
            images[:, 1] = data[:, 1024:2048].reshape(10000, 32, 32) / 255
            images[:, 2] = data[:, 2048:].reshape(10000, 32, 32) / 255
            images = images.transpose((0, 2, 3, 1)) # 10000x32x32x3
            batches.append((images, labels))

    return batches, classes

batches, classes = load_cifar_dataset("../data/cifar10")

n = np.random.randint(0, 10000)
plt.title(classes[batches[0][1][n]])
plt.imshow(batches[0][0][n])
plt.show()