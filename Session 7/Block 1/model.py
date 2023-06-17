import torch
import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

def plot_data(train_loader):
  batch_data, batch_label = next(iter(train_loader)) 

  fig = plt.figure()

  for i in range(12):
      plt.subplot(3,4,i+1)
      plt.tight_layout()
      plt.imshow(batch_data[i].squeeze(0), cmap='gray')
      plt.title(batch_label[i].item())
      plt.xticks([])
      plt.yticks([])

# Train data transformations
train_transforms = transforms.Compose([
    transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
    transforms.Resize((28, 28)),
    transforms.RandomRotation((-15., 15.), fill=0),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    ])

# Test data transformations
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

import torch
import torch.nn as nn

class Net(nn.Module):
  def __init__(self):
      super(Net, self).__init__()
      self.conv1 = nn.Conv2d(1, 16, 3, padding=0) 
      self.bn1 = nn.BatchNorm2d(16)
      self.drop1 = nn.Dropout(0.05)
      
      self.conv2 = nn.Conv2d(16, 32, 3, padding=0)
      self.bn2 = nn.BatchNorm2d(32)
      self.drop2 = nn.Dropout(0.05)
      
      self.conv3 = nn.Conv2d(32, 8, 1, padding=0)
      self.pool3 = nn.MaxPool2d(2, 2)

      self.conv4 = nn.Conv2d(8, 16, 3, padding=0)
      self.bn4 = nn.BatchNorm2d(16)
      self.drop4 = nn.Dropout(0.05)
      
      self.conv5 = nn.Conv2d(16, 32, 3, padding=0)
      self.bn5 = nn.BatchNorm2d(32)
      self.drop5 = nn.Dropout(0.05)
      
      self.conv6 = nn.Conv2d(32, 16, 1, padding=0)
      self.pool6 = nn.MaxPool2d(2, 2)

      self.conv7 = nn.Conv2d(16, 32, 3, padding=0)
      
      self.pool8 = nn.AvgPool2d(2,2)
      
      self.fc1 = nn.Linear(32, 10)
      
  def forward(self, x):
      x = self.drop1(self.bn1(F.relu(self.conv1(x))))
      x = self.drop2(self.bn2(F.relu(self.conv2(x))))
      x = self.pool3(self.conv3(x))
      x = self.drop4(self.bn4(F.relu(self.conv4(x))))
      x = self.drop5(self.bn5(F.relu(self.conv5(x))))
      x = self.pool6(self.conv6(x))
      x = F.relu(self.conv7(x))
      x = self.pool8(x)
      
      x = x.view(x.size(0), -1)
      x = self.fc1(x)
      return F.log_softmax(x, dim=1)

    
# Data to plot accuracy and loss graphs
train_losses = []
test_losses = []
train_acc = []
test_acc = []

test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}

def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def train(model, device, train_loader, optimizer, criterion):
  model.train()
  pbar = tqdm(train_loader)

  train_loss = 0
  correct = 0
  processed = 0

  for batch_idx, (data, target) in enumerate(pbar):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()

    # Predict
    pred = model(data)

    # Calculate loss
    loss = criterion(pred, target)
    train_loss+=loss.item()

    # Backpropagation
    loss.backward()
    optimizer.step()
    
    correct += GetCorrectPredCount(pred, target)
    processed += len(data)

    pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

  train_acc.append(100*correct/processed)
  train_losses.append(train_loss/len(train_loader))

def test(model, device, test_loader, criterion):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss

            correct += GetCorrectPredCount(output, target)


    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
     