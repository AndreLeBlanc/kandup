import math
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.modules.conv import _ConvNd
from torch.nn import functional as F
import cmath
from torch.nn.modules.utils import _pair
from torch import linalg as la

class Zernike(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, device="cuda", stride=1,
                 padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        super(Zernike, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False, _pair(0), groups, bias, padding_mode)
        self.is_calculated = False
        self.device = device
        self.weights = torch.empty(self.weight.shape, requires_grad=False)#.to(self.device)

    def radialDist(self, filtSide, center):
        radials = torch.zeros(filtSide, filtSide)
        for i in range(0, filtSide):
            for j in range(0, filtSide):
                a = abs(i-center)
                b = abs(j-center)
                dist = math.sqrt(a*a + b*b)
                if dist <= center+0.2:
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

    def expo(self, r, rep, filtSide, center, phi):
        for i in range(0, filtSide):
            for j in range(0, filtSide):
                if r[i, j] > 0:
                    localPhi = phi  + math.atan(abs(j-center)/(abs(i-center)+0.001))
                    comp = cmath.exp(rep*localPhi*1j)
                    r[i, j] *= (comp.imag + comp.real)
        return r

    def forward(self, input_tensor):
        order = 0
        rep = 0
        filtSide = self.weights.data.shape[3]
        center = filtSide/2-0.5
        radials = self.radialDist(filtSide, center)
        phi = 0
        g = torch.zeros(self.weights.shape)
        for i in range(self.out_channels):
           for j in range(self.in_channels):
                if rep == order:
                   if order < 6:
                       order += 1
                       rep = 0
                if order == 6:
                   order = 1
                   rep = 0
                   phi += (math.pi)/math.ceil((self.in_channels*self.out_channels)/20-1)
                else:
                   rep += 1

                r = self.polys(radials, filtSide, order, rep)
                exponents = self.expo(r, rep, filtSide, center, phi)
                norm = la.norm(exponents)
                normalized = torch.zeros(self.kernel_size, requires_grad=True)
                normalized.data = normalized.data + torch.div(exponents, norm)
                g[i, j] = normalized
        self.weights = g
        return F.conv2d(input_tensor, self.weights, self.bias, self.stride, self.padding, self.dilation, self.groups)
