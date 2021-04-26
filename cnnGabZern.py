import torch
import torch.nn as nn
import torch.nn.functional as F
from GaborNet import GaborConv2d

class Cnn(nn.Module):
    def __init__(self):
       super(Cnn,self).__init__()

       self.layer1 = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3, padding=0,stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

       self.layer2 = nn.Sequential(
            nn.Conv2d(16,32, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
            )

       self.layer3 = nn.Sequential(
            nn.Conv2d(32,64, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

       self.fc1 = nn.Linear(3*3*64,10)
       self.dropout = nn.Dropout(0.5)
       self.fc2 = nn.Linear(10,2)
       self.relu = nn.ReLU()


    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0),-1)
        print(out.shape)
        out = self.relu(self.fc1(out))
        print(out.shape)
        out = self.fc2(out)
        print(out.shape)
        return out

class Gabor(nn.Module):
   def __init__(self):
      super(Gabor, self).__init__()
      self.g0 = GaborConv2d(in_channels=3, out_channels=16, kernel_size=(11, 11), stride=2)
      self.c1 = nn.Conv2d(16, 64, (3,3), stride=2)
      self.fc1 = nn.Linear(9216, 10)
      self.fc2 = nn.Linear(10, 2)

   def forward(self, x):
       x = F.leaky_relu(self.g0(x))
       pool = nn.MaxPool2d(2)
       x = pool(x)
       x = F.leaky_relu(self.c1(x))
       x = pool(x)
       x = x.view(x.size(0), -1)
       x = F.leaky_relu(self.fc1(x))
       x = self.fc2(x)
       return x
