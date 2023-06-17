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

dropout_value = .01
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels= 32,kernel_size= (3, 3), padding=0, bias=False),    #output_image = 26, RF=3
            nn.ReLU(),
            nn.BatchNorm2d(num_features=32),
            nn.Dropout2d(dropout_value)
            )
        
        #Transition BLOCK 1
        self.trans1 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels= 8,kernel_size= (1, 1), padding=0, bias=False),    #output_image = 26, RF=3
            nn.ReLU(),
            nn.BatchNorm2d(num_features=8),
            nn.Dropout2d(dropout_value)
        )

        #CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8,out_channels= 10,kernel_size= (3, 3), padding=0, bias=False),    #output_image = 24, RF=5
            nn.ReLU(),
            nn.BatchNorm2d(num_features=10),
            nn.Dropout2d(dropout_value)
        )       
        self.pool1 = nn.MaxPool2d(2, 2)    #output_image = 12, RF=6


        #CONVOLUTION BLOCK 2
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10,out_channels=16 ,kernel_size= (3, 3), padding=1, bias=False),    #output_image = 10, RF=10
            nn.ReLU(),
            nn.BatchNorm2d(num_features=16),
            nn.Dropout2d(dropout_value) 
        )
              
        #TRANSITION BLOCK 2 
        self.trans2 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels= 10,kernel_size= (1, 1), padding=0, bias=False),    #output_image = 10, RF=10
            nn.ReLU(),
            nn.BatchNorm2d(num_features=10),
            nn.Dropout2d(dropout_value)
        )

        #CONVOLUTION BLOCK 3
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10,out_channels= 10,kernel_size= (3, 3), padding=0, bias=False),    #output_image = 8, RF=14
            nn.ReLU(),
            nn.BatchNorm2d(num_features=10),
            nn.Dropout2d(dropout_value)
        )       
   
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10,out_channels= 16,kernel_size= (3, 3), padding=0, bias=False),    #output_image = 8, RF=20
            nn.ReLU(),
            nn.BatchNorm2d(num_features=16),
            nn.Dropout2d(dropout_value)
        )
        
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels= 16,kernel_size= (3, 3), padding=0, bias=False),    #output_image = 6, RF=24
            nn.ReLU(),
            nn.BatchNorm2d(num_features=16),
            nn.Dropout2d(dropout_value)
        )

    
        #GAP Layer
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6),
            
        )

        # FC layer
        self.trans3 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels= 10,kernel_size= (1, 1), padding=0, bias=False),    #output_image = 6, RF=28
            #nn.ReLU()  NEVER!!!!
            #nn.BatchNorm2d(num_features=10)     NEVER!!!!
            #nn.Dropout2d(dropout_value)    NEVER!!!!
        )
           

    def forward(self, x):
      x = self.convblock1(x)
      x = self.trans1(x)
      x = self.convblock2(x)      
      x = self.pool1(x)
      x = self.convblock3(x)
      x = self.trans2(x)
      x = self.convblock4(x)      
      x = self.convblock5(x)
      x = self.convblock6(x)     
      x = self.gap(x)
      x =self.trans3(x)
      x = x.view(-1, 10)
      return F.log_softmax(x, dim=-1)
    
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
     