import torch
import torch.nn as nn
import math

def gaussian_kernel2d(ks, sigma, channels = 3):
    x = torch.arange(ks)
    x = x.repeat(ks).view(ks, ks)
    y = x.t()
    xy = torch.stack([x, y], dim=-1)

    mean = (ks - 1)/2.
    variance = sigma**2.

    kernel = torch.exp(-torch.sum((xy - mean)**2., dim=-1) / (2*variance))
    kernel *= 1./(2.*math.pi*variance)
    kernel /= torch.sum(kernel)
    return kernel.view(1, 1, ks, ks).repeat(channels, 1, 1, 1)

def gaussian_conv2d(ks, sigma, channels = 3):
    conv = nn.Conv2d(channels, channels, ks, bias=False, padding=(ks - 1)//2, groups=channels)
    conv.weight.data = gaussian_kernel2d(ks, sigma, channels)
    conv.weight.requires_grad = False
    return conv

class XDOG(nn.Module):
    def __init__(self, k = 4.5, sigma = 0.4, gamma = 0.95, thresh = -0.5, phi = 1e9, ks=25):
        super().__init__()
        self.gauss1 = gaussian_conv2d(ks, sigma, 1)
        self.gauss2 = gaussian_conv2d(ks, gamma*k, 1)
        self.gam,  self.phi = gamma, phi
        self.thresh = nn.Threshold(thresh, 1)
        
    def forward(self, x):
        x = self.gauss1(x) - self.gam*self.gauss2(x)
        x = self.thresh(x)
        x = 1 + torch.tanh(self.phi*x)
        return x
