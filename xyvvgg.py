from math import pi, tau

import fastai.vision.all as fv
import torch
import torch.nn as nn

from layers import *
from colors import rgb2hsv

class RGB2XYV(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x):
        return rgb2xyv(x)

def get_xyvvgg(pretrained=None):
    f0,f1,f2,f3,f4,f5 = 48,64,96,128,192,256

    baseModel = nn.Sequential(
        RGB2XYV(),
        nn.BatchNorm2d(3),
        conv2d(3,f0,k=6,s=2),
        ResBlock(f0,f1),
        nn.MaxPool2d(2),
        *abc(f1,f1),
        ResBlock(f1),
        ResBlock(f1,f2),
        nn.MaxPool2d(2),
        *acb(f2,f3),
        ResBlock(f3),
        *acb(f3,f4,init='id'),
        ResBlock(f4),
        nn.MaxPool2d(2),
        *acb(f4,f4),
        ResBlock(f4,g=2),
        ResBlock(f4),
        ResBlock(f4,f5,s=2),
        ResBlock(f5,g=2),
        ResBlock(f5),
        nn.AdaptiveAvgPool2d(1),
        fv.Flatten(),
        nn.CELU(),
        nn.Linear(f5,1000)
    )
    if pretrained:
        baseModel.load_state_dict(torch.load(pretrained))
    return baseModel