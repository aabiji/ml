%matplotlib inline
from IPython.display import clear_output
from IPython import display

import datasets
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt


def init_layer(layer, last_batchnorm=False):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    if isinstance(layer, nn.BatchNorm2d):
        nn.init.zeros_(layer.bias)
        # zero gamma BatchNorm init
        if last_batchnorm:
            nn.init.zeros_(layer.weight)
        else:
            nn.init.ones_(layer.weight)

    if isinstance(layer, nn.BatchNorm2d) and last_batchnorm:
        nn.init.zeros_(layer.weight)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, downsample):
        super().__init__()

        stride1 = 2 if downsample else 1
        out_channels = in_channels * 2 if downsample else in_channels

        self.layers = nn.ModuleList([
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 3, stride=stride1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        ])

        self.projection = None
        if downsample:
            self.projection = nn.Conv2d(in_channels, out_channels, 1, stride=stride1, padding=0)
            init_layer(self.projection)

        for i, layer in enumerate(self.layers):
            init_layer(layer, i == 3)

    def forward(self, in_data):
        prev_layer_out = in_data.clone()
        for layer in self.layers:
            in_data = layer(in_data)

        if self.projection is not None:
            prev_layer_out = self.projection(prev_layer_out)

        return in_data + prev_layer_out


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.in_conv = nn.Conv2d(3, 16, 3, stride=1, padding=1)

        channels = [16, 16, 16, 32, 32, 64]
        self.blocks = nn.ModuleList([
            ResidualBlock(channels[i], i == 2 or i == 4) for i in range(6) ])

        self.pooling = nn.AvgPool2d((8, 8))
        self.out_linear = nn.Linear(64, 10)

        init_layer(self.in_conv)
        init_layer(self.out_linear)

    def forward(self, in_data):
        in_data = self.in_conv(in_data)

        for block in self.blocks:
            in_data = block(in_data)

        in_data = self.pooling(in_data)
        in_data = in_data.reshape(in_data.shape[0], 64)

        in_data = self.out_linear(in_data)
        return in_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

batches, classes = datasets.load_cifar10(device)
training_batches, test_batch = batches[:5], batches[5]

model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
scheduler = StepLR(optimizer, step_size=12, gamma=0.1)
batch_size, subset, epochs = 10000, 128, 100

losses = []
errors = []
fig, ax = plt.subplots()

# Train model
model.train()
for epoch in range(epochs):
    for i, batch in enumerate(training_batches):
        all_imgs, all_labels = batch
        total_loss = 0

        for j in range(0, batch_size, subset):
            imgs, labels = all_imgs[j:j+subset], all_labels[j:j+subset]
            prediction = model(imgs)
            loss = criterion(prediction, labels)
            total_loss += loss.item()
            model.zero_grad()
            loss.backward()
            optimizer.step()

        # Visualize the training loss curve in real time
        losses.append(total_loss / subset)
        clear_output(wait=True)
        ax.clear()
        ax.plot(losses)
        ax.autoscale_view(scalex=True, scaley=True)
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Loss")
        display.display(fig)

        lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}/{epochs}, Batch: {i+1}/5, LR: {lr}")

    scheduler.step()

# Test performance
model.eval()
for j in range(0, batch_size, subset):
    all_imgs, all_labels = test_batch
    imgs, labels = all_imgs[j:j+subset], all_labels[j:j+subset]
    prediction = model(imgs)

    probabilities = torch.softmax(prediction, dim=1)
    answer = torch.argmax(probabilities, dim=1)
    num_correct = (answer == labels).sum().item()
    errors.append(subset - 100 * num_correct / subset)

mean_error = sum(errors) / len(errors)
print(f"Mean test error: {mean_error}%")