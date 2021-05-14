import math
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.modules import Conv2d, Module
import numpy as np
import math
import cmath
from torch import linalg as la

class Zernike(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        padding_mode="zeros",
    ):
        super().__init__()

        self.is_calculated = False
        self.conv_layer = Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.kernel_size = self.conv_layer.kernel_size

        self.weight = Parameter(
            torch.empty(self.conv_layer.weight.shape, requires_grad=True),
            requires_grad=True,
        )
        self.register_parameter("weight", self.weight)

    def forward(self, input_tensor):
        if self.training:
            self.calculate_weights()
            self.is_calculated = False
        if not self.training:
            if not self.is_calculated:
                self.calculate_weights()
                self.is_calculated = True
        return self.conv_layer(input_tensor)

    def radialDist(self, filtSide, center):
        radials = torch.zeros(filtSide, filtSide)
        for i in range(0, filtSide):
            for j in range(0, filtSide):
                a = abs(i-center)
                b = abs(j-center)
                dist = math.sqrt(a*a + b*b)
                if dist <= center+0.01:
                    radials[i, j] = dist
        return radials

    def polys(self, radials, filtSide, order, rep):
        poly = torch.zeros(filtSide, filtSide)
        for i in range(0, filtSide):
            for j in range(0, filtSide):
                if radials[i, j] > 0:
                    for s in range(0, (order-rep)//2+1):
                        top = ((-1)**s * math.factorial(order-s))
                        bot = (math.factorial(s)*math.gamma((order+rep)/2-s+1)*
                               math.gamma((order-rep)/2-s+1))
                        poly[i, j] += (top/bot)*math.pow(radials[i, j], order-2*s)
        return poly

    def expo(self, r, rep, filtSide, center):
        for i in range(0, filtSide):
            for j in range(0, filtSide):
                if r[i, j] > 0:
                    phi = math.atan(abs(j-center)/(abs(i-center)+0.001))
                    r[i, j] *= cmath.exp(rep*phi*1j).real
        return r

    def calculate_weights(self):
        order = 0
        rep = 0
        filtSide = self.conv_layer.weight.data.shape[3]
        center = filtSide/2-0.5
        radials = self.radialDist(filtSide, center)
        for i in range(self.conv_layer.out_channels):
           if rep == order:
               if order < 6:
                   order += 1
                   rep = 0
           else:
               rep += 1
           for j in range(self.conv_layer.in_channels):
                r = self.polys(radials, filtSide, order, rep)
                exponents = self.expo(r, rep, filtSide, center)
                norm = la.norm(exponents)
                normalized = torch.div(exponents, norm)
                self.conv_layer.weight.data[i, j] = normalized
