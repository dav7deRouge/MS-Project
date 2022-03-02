import torch
import torch.nn.functional
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm

def compute_accuracy(model, dataloader):
  correct = 0.0
  n_data = len(dataloader.dataset)
  with torch.no_grad():
    model.eval()
    for data, labels in tqdm(dataloader, desc="Computing accuracy"):
      data = data.cuda()
      labels = labels.cuda()

      output = model(data)
      output = torch.nn.functional.softmax(output, dim=1)
      correct = correct + torch.sum(output.argmax(dim=1) == labels) #labels.argmax(dim=1)
  accuracy = (correct / n_data) * 100
  return accuracy

def compute_loss_on_batch(model, dataloader):
  running_loss = 0.0
  with torch.no_grad():
    model.eval()
    for data, labels in dataloader:
      data = data.cuda()
      labels = labels.cuda()

      output = model(data)
      loss = torch.nn.functional.cross_entropy(output, labels) #f.binary_cross_entropy_with_logits(output, labels.float())
      running_loss += loss.item()
  return running_loss/len(dataloader)