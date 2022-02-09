import torch
import numpy as np

# Domain adaptation modules: Gradient Reversal Layer, Discriminator
class ReverseLayerF(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, alpha):
    ctx.alpha = alpha

    return x.view_as(x)

  @staticmethod
  def backward(ctx, grad_output):
    output = grad_output.neg() * ctx.alpha

    return output, None

class Discriminator(torch.nn.Module):
  def __init__(self, dim):
    super(Discriminator, self).__init__()
    self.discriminator = torch.nn.Sequential(
        torch.nn.Linear(dim, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, out_features=2)
    )

  def forward(self, input_feature, alpha=1):
    reversed_input = ReverseLayerF.apply(input_feature, alpha)
    x = self.discriminator(reversed_input)
    return x

# Classifier
class Classifier(torch.nn.Module):
  def __init__(self, dim):
    super(Classifier, self).__init__()
    self.classifier = torch.nn.Sequential(
        torch.nn.Linear(dim, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 3)
    )

  def forward(self, x):
    x = self.classifier(x)
    return x