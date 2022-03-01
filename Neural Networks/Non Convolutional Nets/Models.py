from copy import deepcopy
import torch 
import numpy as np
from ..Modules import Classifier, Discriminator

# Shallow Neural Network
class Architettura1(torch.nn.Module):
  def __init__(self, input_size, output_size, hidden_nodes, hidden_func):
    super(Architettura1, self).__init__()
    self.block = torch.nn.Sequential(
        torch.nn.Linear(input_size, hidden_nodes),
        hidden_func,
        torch.nn.Linear(hidden_nodes, output_size)
    )

  def forward(self, input):
    return self.block(input)

# Deep Neural Network
class Architettura2(torch.nn.Module):
  def __init__(self, input_size, output_size, nodes_hidden_layers, nodes_classifier):
    super(Architettura2, self).__init__()
    self.fe = torch.nn.Sequential(
        torch.nn.Linear(input_size, nodes_hidden_layers[0]),
        torch.nn.ReLU(),
        torch.nn.Linear(nodes_hidden_layers[0], nodes_hidden_layers[1]),
        torch.nn.ReLU()
    )
    
    
    self.classifier = torch.nn.Sequential(
        torch.nn.Linear(nodes_hidden_layers[1], nodes_classifier),
        torch.nn.ReLU(),
        torch.nn.Linear(nodes_classifier, output_size),
        torch.nn.ReLU()
    )
    
  def forward(self, input):
    out = self.fe(input)
    return self.classifier(out)

# DA Deep Neural Network
class Architettura3(torch.nn.Module):
  def __init__(self, input_size, output_size, nodes_hidden_layers, nodes_classifier):
    super(Architettura3, self).__init__()
    self.fe = torch.nn.Sequential(
        torch.nn.Linear(input_size, nodes_hidden_layers[0]),
        torch.nn.ReLU(),
        torch.nn.Linear(nodes_hidden_layers[0], nodes_hidden_layers[1]),
        torch.nn.ReLU()
    )
    
    self.classifier = Classifier(nodes_hidden_layers[1])

    self.discriminator = Discriminator(nodes_hidden_layers[1])
    

  def forward(self, input):
    out = self.fe(input)
    return self.classifier(out)