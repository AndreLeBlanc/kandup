import torch
import torch.nn as nn
import torch.nn.functional as F
from GaborNet import GaborConv2d
from myFilt import Zernike

class oneConv(nn.Module):
   def __init__(self, kernelSize, fLayer):
      super(oneConv, self).__init__()
      fcIn = ((220-kernelSize+1)//2-2)//2//2//2
      fcIn = fcIn*fcIn*64
      self.c0 = nn.Conv2d(in_channels=3, out_channels=16,
                            kernel_size=kernelSize, stride=2)
      self.g0 = GaborConv2d(in_channels=3, out_channels=16,
                            kernel_size=kernelSize, stride=2)
      self.z0 = Zernike(in_channels=3, out_channels=16,
                            kernel_size=kernelSize, stride=2)
      self.c1 = nn.Conv2d(16, 64, (3,3), stride=2)
      self.fc1 = nn.Linear(fcIn, 10)
      self.fc2 = nn.Linear(10, 2)
      self.fLayer = fLayer

   def forward(self, x):
       if self.fLayer == "Conv2d":
           x = F.leaky_relu(self.c0(x))
       elif self.fLayer == "Zern":
           x = F.leaky_relu(self.z0(x))
       else:
           x = F.leaky_relu(self.g0(x))
       pool = nn.MaxPool2d(2)
       x = pool(x)
       x = F.leaky_relu(self.c1(x))
       x = pool(x)
       x = x.view(x.size(0), -1)
       x = F.leaky_relu(self.fc1(x))
       x = self.fc2(x)
       return x

class threeConv(nn.Module):
   def __init__(self, kernelSize, fType):
      super(threeConv, self).__init__()
      fcIn = ((((220-kernelSize+1)//4-2)//2-2)//2-2)//2
      fcIn = fcIn*fcIn*256
      self.g0 = GaborConv2d(in_channels=3, out_channels=96,
                            kernel_size=(kernelSize, kernelSize), stride=2)
      self.c0 = nn.Conv2d(in_channels=3, out_channels=96,
                            kernel_size=(kernelSize, kernelSize), stride=2)
      self.c1 = nn.Conv2d(96, 256, (3,3), stride=1)
      self.c2 = nn.Conv2d(256, 384, (3,3), stride=1)
      self.c3 = nn.Conv2d(384, 256, (3,3), stride=1)
      self.fc1 = nn.Linear(fcIn, 128)
      self.fc2 = nn.Linear(128, 2)
      self.fType = fType

   def forward(self, x):
       if self.fType == "Conv2d":
           x = F.leaky_relu(self.c0(x))
       else:
           x = F.leaky_relu(self.g0(x))
       pool = nn.MaxPool2d(2)
       x = pool(x)
       x = F.leaky_relu(self.c1(x))
       x = pool(x)
       x = F.leaky_relu(self.c2(x))
       x = pool(x)
       x = F.leaky_relu(self.c3(x))
       x = pool(x)
       x = x.view(x.size(0), -1)
       x = F.leaky_relu(self.fc1(x))
       x = self.fc2(x)
       return x
