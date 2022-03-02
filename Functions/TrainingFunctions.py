import torch
import torch.nn.functional 
from torch.utils.data import Dataset, DataLoader, TensorDataset

def standard_train(net, optimizer, train_dataloader, val_dataloader, epochs):
  '''
  Standard training function for neural networks.
  '''
  
  best_net = None
  best_val_loss = float("inf")
  val_loss = float("inf")

  net.train()
  for epoch in range(epochs):
    #pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
    for data, labels in train_dataloader:
      data = data.cuda()
      labels = labels.cuda()
      # === FORWARD STEP === #
      pred = net(data)
      train_loss = f.cross_entropy(pred, labels)
      # === BACKWARD STEP === 
      train_loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      #pbar.set_postfix({'train loss': train_loss.item(), 'validation loss': val_loss})
    # === EVALUATION ===
    val_loss = compute_loss_on_batch(net, val_dataloader)
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      best_net = deepcopy(net)
  net = best_net

def dann_train(net, optimizer, train_dataloader, val_dataloader, epochs, alpha, source_only=False):
  '''
  Domain Adaptation training function. 
  Parameters:
  - net : the neural network to be trained
  - optimizer : optimizer algorithm
  - train_dataloader: dataloader containing the elements (x), class labels (y), domain labels (domains) (Training set)
  - val_dataloader:  dataloader containing the elements(x), class labels (y) (Validation set)
  - epochs: number of epochs of the training session
  - alpha: Gradient Reversal Layer parameter (Domain Adversarial Training)
  - source_only: boolean argument used to exclude target labels
  '''

  best_net = None
  best_val_loss = float("inf")
  val_loss = float("inf")

  net.train()
  for epoch in tqdm(range(epochs)):
    #pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
    for x, y, domains in train_dataloader:
      if source_only:
        source_x, source_y = (x[np.where(domains == 0)[0]]).cuda(), (y[np.where(domains == 0)[0]]).cuda()  
      x = x.cuda()
      y = y.cuda()
      domains = domains.cuda() 
      # === FORWARD STEP - DISCRIMINATION ===
      out = net.fe(x)
      pred_d = net.discriminator(out, alpha)
      loss_d = torch.nn.functional.cross_entropy(pred_d, domains) 
      # === FORWARD STEP - CLASSIFICATION ===
      if source_only:
        pred_l = net(source_x)
        loss_l = torch.nn.functional.cross_entropy(pred_l, source_y)
      else:
        pred_l = net.classifier(out)
        loss_l = torch.nn.functional.cross_entropy(pred_l, y)
      loss_train = loss_l + loss_d
      # === BACKWARD STEP === 
      loss_train.backward()
      optimizer.step()
      optimizer.zero_grad()
      #pbar.set_postfix({'train loss': loss_train.item(), 'validation loss': val_loss})
    # === EVALUATION ===
    val_loss = compute_loss_on_batch(net, val_dataloader)
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      best_net = deepcopy(net)
  net = best_net