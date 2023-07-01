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
import torch.nn.functional as F

class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(DepthwiseSeparableBlock, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.depthwise(x)
        out = F.relu(self.bn(self.pointwise(out)))
        return out

class ResNetLikeModel(nn.Module):
    def __init__(self):
        super(ResNetLikeModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.block1_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.block1_bn1 = nn.BatchNorm2d(64)
        self.block1_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.block1_bn2 = nn.BatchNorm2d(64)
        self.block1 = DepthwiseSeparableBlock(64, 64, stride=1)

        self.block2_conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.block2_bn1 = nn.BatchNorm2d(128)
        self.block2_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.block2_bn2 = nn.BatchNorm2d(128)
        self.block2 = DepthwiseSeparableBlock(128, 128, stride=1)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))

        residual = out
        out = F.relu(self.block1_bn1(self.block1_conv1(out)))
        out = F.relu(self.block1_bn2(self.block1_conv2(out)))
        out = self.block1(out)
        # out += residual

        residual = out
        out = F.relu(self.block2_bn1(self.block2_conv1(out)))
        out = F.relu(self.block2_bn2(self.block2_conv2(out)))
        out = self.block2(out)
        # out += residual

        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


    
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
     