# Small ResNet model implemented from scratch.
import numpy as np
import matplotlib.pyplot as plt
import datasets

batches, classes = datasets.load_cifar10("../data/cifar10")
imgs, labels = batches[0]