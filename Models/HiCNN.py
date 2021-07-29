# Code was taken from http://dna.cs.miami.edu/HiCNN2/
import torch
import torch.nn as nn


class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()
    self.conv1 = nn.Conv2d(1, 8, 13)
    self.conv2 = nn.Conv2d(8, 1, 1)
    self.conv3 = nn.Conv2d(1, 128, 3, padding=1, bias=False)
    self.conv4R = nn.Conv2d(128, 128, 3, padding=1, bias=False)
    self.conv5 = nn.Conv2d(128, 1, 3, padding=1, bias=False)
    self.relu = nn.ReLU(inplace=True)
    # He initialization
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

  def forward(self, x):
    x = self.relu(self.conv1(x))
    x = self.relu(self.conv2(x))
    residual = x
    x2 = self.conv3(x)
    out = x2
    for _ in range(25):
      out = self.conv4R(self.relu(self.conv4R(self.relu(out))))
      out = torch.add(out, x2)

    out = self.conv5(self.relu(out))
    out = torch.add(out, residual)
    return out
