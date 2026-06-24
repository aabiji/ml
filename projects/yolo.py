import torch
import torch.nn as nn

class Network(nn.Module):
  def __init__(self):
    super().__init__()

    self.layers = nn.ModuleList([
      nn.Conv2d(3, 64, 7, stride=2, padding=3), # 224 x 224 x 64
      nn.MaxPool2d(2, stride=2), # 112 x 112 x 64
      nn.Conv2d(64, 192, 3, stride=1, padding=1), # 112 x 112 x 192
      nn.MaxPool2d(2, stride=2), # 56 x 56 x 192
      nn.Conv2d(192, 128, 1, stride=1, padding=0), # 56 x 56 x 128
      nn.Conv2d(128, 256, 3, stride=1, padding=1), # 56 x 56 x 256
      nn.Conv2d(256, 256, 1, stride=1, padding=0), # 56 x 56 x 256
      nn.Conv2d(256, 512, 3, stride=1, padding=1), # 56 x 56 x 512
      nn.MaxPool2d(2, stride=2), # 28 x 28 x 512
      nn.Conv2d(512, 256, 1, stride=1, padding=0), # 28 x 28 x 256
      nn.Conv2d(256, 512, 3, stride=1, padding=1), # 28 x 28 x 512
      nn.Conv2d(512, 256, 1, stride=1, padding=0), # 28 x 28 x 256
      nn.Conv2d(256, 512, 3, stride=1, padding=1), # 28 x 28 x 512
      nn.Conv2d(512, 256, 1, stride=1, padding=0), # 28 x 28 x 256
      nn.Conv2d(256, 512, 3, stride=1, padding=1), # 28 x 28 x 512
      nn.Conv2d(512, 256, 1, stride=1, padding=0), # 28 x 28 x 256
      nn.Conv2d(256, 512, 3, stride=1, padding=1), # 28 x 28 x 512
      nn.Conv2d(512, 512, 1, stride=1, padding=0), # 28 x 28 x 512
      nn.Conv2d(512, 1024,3, stride=1, padding=1), # 28 x 28 x 1024
      nn.MaxPool2d(2, stride=2), # 14 x 14 x 1024
      nn.Conv2d(1024,512, 1, stride=1, padding=0), # 14 x 14 x 512
      nn.Conv2d(512, 1024,3, stride=1, padding=1), # 14 x 14 x 1024
      nn.Conv2d(1024,512, 1, stride=1, padding=0), # 14 x 14 x 512
      nn.Conv2d(512, 1024,3, stride=1, padding=1), # 14 x 14 x 1024
      nn.Conv2d(1024,1024,3, stride=1, padding=1), # 14 x 14 x 1024
      nn.Conv2d(1024,1024,3, stride=2, padding=1), # 7 x 7 x 1024
      nn.Conv2d(1024,1024,3, stride=1, padding=1), # 7 x 7 x 1024
      nn.Conv2d(1024,1024,3, stride=1, padding=1), # 7 x 7 x 1024
      nn.Linear(7 * 7 * 1024, 4096),
      nn.Linear(4096, 7 * 7 * 30)
    ])


