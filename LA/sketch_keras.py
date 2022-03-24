# Code based on https://github.com/higumax/sketchKeras-pytorch
import math
from pathlib import Path
import torch
import torch.nn as nn

BASE_DIR = Path(__file__).resolve().parent

class SketchKerasModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.downblock_1 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(1, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32, eps=1e-3, momentum=0),
            nn.ReLU(),
        )
        self.downblock_2 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64, eps=1e-3, momentum=0),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64, eps=1e-3, momentum=0),
            nn.ReLU(),
        )
        self.downblock_3 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0),
            nn.ReLU(),
        )
        self.downblock_4 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0),
            nn.ReLU(),
        )
        self.downblock_5 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, kernel_size=4, stride=2),
            nn.BatchNorm2d(512, eps=1e-3, momentum=0),
            nn.ReLU(),
        )
        self.downblock_6 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, kernel_size=3, stride=1),
            nn.BatchNorm2d(512, eps=1e-3, momentum=0),
            nn.ReLU(),
        )

        self.upblock_1 = nn.Sequential(
            nn.Upsample((64, 64)),
            nn.ReflectionPad2d((1, 2, 1, 2)),
            nn.Conv2d(1024, 512, kernel_size=4, stride=1),
            nn.BatchNorm2d(512, eps=1e-3, momentum=0),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0),
            nn.ReLU(),
        )

        self.upblock_2 = nn.Sequential(
            nn.Upsample((128, 128)),
            nn.ReflectionPad2d((1, 2, 1, 2)),
            nn.Conv2d(512, 256, kernel_size=4, stride=1),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0),
            nn.ReLU(),
        )

        self.upblock_3 = nn.Sequential(
            nn.Upsample((256, 256)),
            nn.ReflectionPad2d((1, 2, 1, 2)),
            nn.Conv2d(256, 128, kernel_size=4, stride=1),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64, eps=1e-3, momentum=0),
            nn.ReLU(),
        )

        self.upblock_4 = nn.Sequential(
            nn.Upsample((512, 512)),
            nn.ReflectionPad2d((1, 2, 1, 2)),
            nn.Conv2d(128, 64, kernel_size=4, stride=1),
            nn.BatchNorm2d(64, eps=1e-3, momentum=0),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32, eps=1e-3, momentum=0),
            nn.ReLU(),
        )

        self.last_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.last_conv = nn.Conv2d(64, 1, kernel_size=3, stride=1)

    def forward(self, x):
        d1 = self.downblock_1(x)
        d2 = self.downblock_2(d1)
        d3 = self.downblock_3(d2)
        d4 = self.downblock_4(d3)
        d5 = self.downblock_5(d4)
        d6 = self.downblock_6(d5)

        u1 = torch.cat((d5, d6), dim=1)
        u1 = self.upblock_1(u1)
        u2 = torch.cat((d4, u1), dim=1)
        u2 = self.upblock_2(u2)
        u3 = torch.cat((d3, u2), dim=1)
        u3 = self.upblock_3(u3)
        u4 = torch.cat((d2, u3), dim=1)
        u4 = self.upblock_4(u4)
        u5 = torch.cat((d1, u4), dim=1)

        out = self.last_conv(self.last_pad(u5))

        return out

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
    return conv

# Recieves color image
class SketchKeras(nn.Module):
    def __init__(self, gauss_ks=25, gauss_sigma = 3, thresh=0.1):
        super().__init__()
        self.model = SketchKerasModel()
        self.model.load_state_dict(torch.load(BASE_DIR/"models"/"sketchKeras.pth"))
        self.gauss = gaussian_conv2d(gauss_ks, gauss_sigma)
        self.thresh = nn.Threshold(thresh, 0)
    
    def forward(self, x):
        x = x - self.gauss(x)
        x =  x / x.max()
        x = 0.2989 * x[:,0] + 0.5870 * x[:,1] + 0.1140 * x[:,2] # TO BW
        
        pred = self.model(x[:,None])
        
        pred = self.thresh(pred)
        pred = pred.clamp_max(1)
        pred = 1 - pred
        return pred