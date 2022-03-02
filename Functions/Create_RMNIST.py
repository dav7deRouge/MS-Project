import numpy as np
import torch
from torchvision.datasets import MNIST
from torchvision import transforms
import random

def rotate_dataset(d, rotation):
    '''
    Function to rotate MNIST images.
    Parameters:
    - d : dataset
    - rotation : angle used to rotate images
    '''

    result = torch.FloatTensor(d.size(0), 28, 28)
    tensor = transforms.ToTensor()

    for i in range(d.size(0)):
        img = Image.fromarray(d[i].numpy(), mode='L')
        result[i] = tensor(img.rotate(rotation))
    return result

# MNIST data
data = MNIST(root = '.', train = True, download = True)
train_x, train_y = data.data, data.targets

# Subsampling dataset
train = list(zip(train_x, train_y))
selected = random.sample(train, 10000)
train_x = torch.stack([d for (d, l) in selected])
train_y = torch.stack([l for (d, l) in selected])

# Build R-MNIST
angles = [15, 30, 45, 60, 75]
all_domain_train = torch.empty((70000, 28, 28))
all_domain_train[0:10000, :, :] = train_x[:, :, :]
i = 1
for a in angles:
  all_domain_train[i*10000:(i*10000)+10000, :, :] = rotate_dataset(train_x, a)
  i += 1
print(f"Dataset shape = {all_domain_train.shape}")

# Flatten the arrays
all_domain_train = torch.reshape(all_domain_train, (all_domain_train.shape[0], -1))
print(f"New dataset shape = {all_domain_train.shape}")