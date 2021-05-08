import math
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.modules import Conv2d, Module
import numpy as np
import math

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
                if dist <= math.sqrt(center*center+0.25):
                    radials[i, j] = math.sqrt(a*a + b*b)
        return radials

    def polys(self, radials, filtSide, n, m):
        poly = torch.zeros(filtSide, filtSide)
        for i in range(0, filtSide):
            for j in range(0, filtSide):
                if radials[i, j] > 0:
                    for s in range(0, (n-m)//2):
                        top=((-1)**s * math.factorial(n-s))
                        bot =(math.factorial(s)*math.gamma((n+m)/2-s+1)*math.gamma((n-m)/2-s)+1)
                        poly[i, j] += (top/bot)*math.pow(radials[i, j], n-2*s)
        return poly

    def expo(self, r, n, filtSide, center):
        for i in range(0, filtSide):
            for j in range(0, filtSide):
                if r[i, j] > 0:
                    phi = math.atan(abs(j-center)/(abs(i-center)+0.001))
                    r[i, j] *= 2.71828**(n*phi*1j).real
        return r

    def calculate_weights(self):
        n = 2
        m=0
        filtSide = self.conv_layer.weight.data.shape[3]
        center = filtSide/2-0.5
        for i in range(self.conv_layer.out_channels):
            for j in range(self.conv_layer.in_channels):
                if m == n:
                    n += 1
                    m = 0
                else:
                    m += 1
                radials = self.radialDist(filtSide, center)
                r = self.polys(radials, filtSide, n, m)
                exponents = self.expo(r, n, filtSide, center)
                print(exponents)
                self.conv_layer.weight.data[i, j] = exponents
