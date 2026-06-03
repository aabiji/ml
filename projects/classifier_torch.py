import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import datasets

using_iris = False
x_train, y_train, x_test, y_test = datasets.load_iris() if using_iris else datasets.load_mnist(
    "../data/mnist/train-images.idx3-ubyte",
    "../data/mnist/train-labels.idx1-ubyte",
    "../data/mnist/t10k-images.idx3-ubyte",
    "../data/mnist/t10k-labels.idx1-ubyte"
)

d_i, d_k, d_o, epochs = 784, 100, 10, 100
if using_iris:
    d_i, d_k, d_o, epochs = 4, 10, 3, 100

batch_size = int(x_train.shape[0] / 10)
batches_per_epoch = x_train.shape[0] / batch_size

model = torch.nn.Sequential(
    # Input layer
    torch.nn.Linear(d_i, d_k),
    torch.nn.ReLU(),
    # 3 hidden layers
    torch.nn.Linear(d_k, d_k),
    torch.nn.ReLU(),
    torch.nn.Linear(d_k, d_k),
    torch.nn.ReLU(),
    torch.nn.Linear(d_k, d_k),
    torch.nn.ReLU(),
    # Output layer
    torch.nn.Linear(d_k, d_o)
)

def init_weights(layer_in):
    if type(layer_in) is nn.Linear:
        init.kaiming_normal_(layer_in.weight)
        init.zeros_(layer_in.bias)
model.apply(init_weights)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = StepLR(optimizer, gamma=0.5, step_size=10)
data_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
accuracies = np.zeros((epochs, 1))

for epoch in range(epochs):
    for i, batch in enumerate(data_loader):
        x_batch, y_batch = batch
        optimizer.zero_grad()

        prediction = model(x_batch)
        loss = loss_function(prediction, y_batch)
        loss.backward()
        optimizer.step()

    num_correct = 0
    prediction = model(x_test)
    loss = loss_function(prediction, y_test)
    max_indices = torch.argmax(prediction, dim=-1)

    if using_iris:
        predicted_class = F.one_hot(max_indices, num_classes=prediction.shape[-1])
        num_correct = (predicted_class == y_test).all(dim=1).sum()
    else:
        num_correct = (max_indices == y_test).sum()
    accuracies[epoch] = 100 * num_correct / y_test.shape[0]

    print(f"Epoch {epoch + 1}/{epochs} | Loss: {loss:.3f} | Accuracy: {accuracies[epoch][0]:.3f}%")
    scheduler.step()

fig, ax = plt.subplots()
x_axis = np.arange(0, epochs, 1)
ax.plot(x_axis, accuracies, color="red", linestyle='-')
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy (%)")
plt.show()