import torch.nn as nn
import torch
import torch.nn.functional as F
from GaborNet import GaborConv2d
from myFilt import Zernike

class oneConv(nn.Module):
   def __init__(self, kernelSize, fLayer, inputs, imSize):
      super(oneConv, self).__init__()
      fcIn = ((imSize-kernelSize+1)//2-2)//2//2//2
      fcIn = fcIn*fcIn*256
      self.c0 = nn.Conv2d(in_channels=inputs, out_channels=96,
                            kernel_size=kernelSize, stride=2)
      self.g0 = GaborConv2d(in_channels=inputs, out_channels=96,
                            kernel_size=kernelSize, stride=2)
      self.z0 = Zernike(in_channels=inputs, out_channels=96,
                            kernel_size=kernelSize, stride=2)
      self.c1 = nn.Conv2d(96, 256, (3,3), stride=2)
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
   def __init__(self, kernelSize, fType, inputs, imSize):
      super(threeConv, self).__init__()
      fcIn = ((((imSize-kernelSize+1)//4-2)//2-2)//2-2)
      fcIn = fcIn*fcIn*128
      self.g0 = GaborConv2d(in_channels=inputs, out_channels=32,
                            kernel_size=kernelSize, stride=1)
      self.c0 = nn.Conv2d(in_channels=inputs, out_channels=32,
                            kernel_size=kernelSize, stride=1)
      self.z0 = Zernike(in_channels=inputs, out_channels=32,
                            kernel_size=kernelSize, stride=1)
      self.c1 = nn.Conv2d(32, 64, (3,3), stride=1)
      self.c2 = nn.Conv2d(64, 128, (3,3), stride=1)
      self.c3 = nn.Conv2d(128, 128, (3,3), stride=1)
      self.c4 = nn.Conv2d(128, 128, (3,3), stride=1)
      self.fc1 = nn.Linear(fcIn, 128)
      self.fc2 = nn.Linear(128, 2)
      self.fType = fType

   def forward(self, x):
       if self.fType == "Conv2d":
           x = F.leaky_relu(self.c0(x))
       elif self.fType == "Zern":
          x = F.leaky_relu(self.z0(x))
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
       x = F.leaky_relu(self.c4(x))
       x = x.view(x.size(0), -1)
       x = F.leaky_relu(self.fc1(x))
       x = self.fc2(x)
       return x


    ################
    ## ALEXnet #####
    ################

class AlexNet(nn.Module):
    def __init__(self, fType, global_params=None):
        super(AlexNet, self).__init__()
        self.fType = fType
        self.c0 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.z0 = Zernike(1, 64, kernel_size=11, stride=4)
        self.g0 = GaborConv2d(1, 64, kernel_size=11, stride=4)

        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 220, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(220, 220, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(220 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """
        x = self.features(inputs)
        return x

    def forward(self, inputs):
        # See note [TorchScript super()]
        if self.fType == "Conv2d":
            x = self.c0(inputs)
        elif self.fType == "Zern":
            x = self.z0(inputs)
        else:
            x = self.g0(inputs)

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
