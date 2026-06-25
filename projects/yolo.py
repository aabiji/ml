# The following is a reimplementation of the original Yolo paper.

import torch
import torch.nn as nn

# input channels, output channels, kernel size, stride, padding
layer_info = [
  (3, 64, 7, 2, 3),    (64,  192, 3, 1, 1), (192, 128, 1, 1, 0),
  (128, 256, 3, 1, 1), (256, 256, 1, 1, 0), (256, 512, 3, 1, 1),
  (512, 256, 1, 1, 0), (256, 512, 3, 1, 1), (512, 256, 1, 1, 0),
  (256, 512, 3, 1, 1), (512, 256, 1, 1, 0), (256, 512, 3, 1, 1),
  (512, 256, 1, 1, 0), (256, 512, 3, 1, 1), (512, 512, 1, 1, 0),
  (512, 1024,3, 1, 1), (1024,512, 1, 1, 0), (512, 1024,3, 1, 1),
  (1024,512, 1, 1, 0), (512, 1024,3, 1, 1), (1024,1024,3, 1, 1),
  (1024,1024,3, 2, 1), (1024,1024,3, 1, 1), (1024,1024,3, 1, 1),
]
maxpool_indices = [1, 3, 8, 19]

class Network(nn.Module):
  def __init__(self):
    super(Network, self).__init__()

    self.layers = nn.ModuleList([])
    for i, l in enumerate(layer_info):
      self.layers.append(nn.Conv2d(l[0], l[1], l[2], stride=l[3], padding=l[4]))
      if i in maxpool_indices:
        self.layers.append(nn.MaxPool(2, stride=2))
      self.layers.append(nn.LeakyReLU(0.1))

    self.layers.append(nn.Linear(7 * 7 * 1024, 4096))
    self.layers.append(nn.LeakyReLU(0.1))
    self.layers.append(nn.Linear(4096, 7 * 7 * 30))

  def forward(self, x):
    for layer in self.layers:
      x = layer(x)
    return x


class SumOfSquares(nn.Module):
  def __init__(self):
    super(SumOfSquares, self).__init__()

  # See section 2.2 of paper, assuming that 'pred' and 'real' are 7x7x30 tensors
  def forward(self, pred, real):
    # Real and predicted (x, y, w, h, c) for first bounding box
    px1, py1, pw1, ph1, pc1 = pred[:, :, 20:25]
    x1,  y1,  w1,  h1,  c1  = real[:, :, 20:25]
    pw1, ph1, w1,  h1 = torch.sqrt(pw1), torch.sqrt(ph1), torch.sqrt(w1), torch.sqrt(h1)

    # Real and predicted (x, y, w, h, c) for second bounding box
    px2, py2, pw2, ph2, pc2 = pred[:, :, 25:30]
    x2,  y2,  w2,  h2,  c2  = real[:, :, 25:30]
    pw2, ph2, w2,  h2 = torch.sqrt(pw2), torch.sqrt(ph2), torch.sqrt(w2), torch.sqrt(h2)

    # Classes and total confidence
    pred_class, real_class = pred[: :, 0:20], real[:, :, 0:20]
    pc = torch.logical_or(pc1 > 0, pc2 > 0)

    a = torch.sum(((pc1 > 0) * ((px1 - x1) ** 2 + (py1 - y1) ** 2)) + \
          ((pc2 > 0) * ((px2 - x2) ** 2 + (py2 - y2) ** 2)))
    b = torch.sum(((pc1 > 0) * ((pw1 - w1) ** 2 + (ph1 - h1) ** 2)) + \
          ((pc2 > 0) * ((pw2 - w2) ** 2 + (ph2 - h2) ** 2)))
    c = torch.sum(((pc1 > 0) * ((pc1 - c1) ** 2)) + ((pc2 > 0) * ((pc2 - c2) ** 2)))
    d = torch.sum(((pc1 == 0) * ((pc1 - c1) ** 2)) + ((pc2 == 0) * ((pc2 - c2) ** 2)))
    e = torch.sum(pc * torch.sum((pred_class - real_class) ** 2, axis=2))
    return 5 * a + 5 * b + c + 0.5 * d + e

def train(model, device, data_loader, optimizer, epochs):
  model.train()

  for i, (batch_x, batch_y) in enumerate(data_loader):
    batch_x, batch_y = device.to(batch_x), device.to(batch_y)
    optimizer.zero_grad()
    output = model(batch_x)
    loss = SumOfSquares(output, batch_y)
    loss.backward()
    optimizer.step()
    print(f"Batch {i + 1}, loss = {loss.item()}")



