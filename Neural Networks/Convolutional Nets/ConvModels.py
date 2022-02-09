from copy import deepcopy
import torch 
import numpy as np
from ..Modules import Classifier, Discriminator

# Core module of Resnet 
class ResNetBlock(torch.nn.Module):
  def __init__(self, in_channels, conv_channels):
    super(ResNetBlock, self).__init__()
    self.transform_input = torch.nn.Identity() if in_channels == conv_channels else torch.nn.Conv1d(in_channels=in_channels, out_channels=conv_channels, kernel_size=1)
    
    self.block = torch.nn.Sequential(
        torch.nn.Conv1d(in_channels=in_channels, out_channels=conv_channels, kernel_size=3, padding=1),
        torch.nn.BatchNorm1d(conv_channels),
        torch.nn.ReLU(),
        torch.nn.Conv1d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=3, padding=1),
        torch.nn.BatchNorm1d(conv_channels)
    )

  def forward(self, input):
    residual = self.transform_input(input)
    return torch.nn.functional.relu_(residual + self.block(input))

# Resnet version with MaxPool layers
class Net(torch.nn.Module):
  def __init__(self, in_channels, conv_channels, num_blocks):
    super(Net, self).__init__()
    self.resblock = torch.nn.ModuleList([ResNetBlock(in_channels, conv_channels), torch.nn.MaxPool1d(2)])
    bs = num_blocks - 1
    if bs > 1:
      out_dim = math.floor(62 / 2**3)
      half = math.floor(bs / 2)
      i = 1
      while i <= bs:
        # Adding maxpool layers in the middle and end
        if i == half or i == bs:
          self.resblock += torch.nn.ModuleList([ResNetBlock(conv_channels, conv_channels), torch.nn.MaxPool1d(2)])
        else:
          self.resblock += torch.nn.ModuleList([ResNetBlock(conv_channels, conv_channels)])
        i += 1
    else:
      out_dim = 31

    self.classifier = Classifier(out_dim*conv_channels)

    self.discriminator = Discriminator(out_dim*conv_channels)

  def forward(self, x):
    for b in self.resblock:
      x = b(x)
    return self.classifier(x.flatten(1))