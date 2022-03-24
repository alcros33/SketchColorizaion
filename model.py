from pathlib import Path
from math import tau,pi
import gc, os
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import fastai.vision.all as fv
from colors import *
from layers import *

class ImageTensorOutput(fv.TensorImage): pass
class PILImageOutput(fv.PILImage): pass
PILImageOutput._tensor_cls = ImageTensorOutput
class ImageTensorInput(fv.TensorImage): pass
class PILImageInput(fv.PILImage): pass
PILImageInput._tensor_cls = ImageTensorInput

class HHSV2RGB(fv.Transform):
    order = 100
    def decodes(self, o:fv.TensorImage):
        if o.shape[1] == 4:
            return hhsv2rgb(o)
        return o
    def decodes(self, o:ImageTensorOutput):
        if o.shape[1] == 4:
            return hhsv2rgb(o)
        return o
    def encodes(self, o:ImageTensorOutput):
        return o
    def encodes(self, o:ImageTensorInput):
        return rgb2hhsv(o)

def ImageBlockHHSV(cls=fv.PILImage):
    return fv.TransformBlock(type_tfms=cls.create, batch_tfms=[fv.IntToFloatTensor, HHSV2RGB])

class HueDecoder(nn.Module):
    def __init__(self, lin_deco, conv_deco):
        super().__init__()
        self.lin_deco = lin_deco
        self.conv_deco = conv_deco
        
    def forward(self, z):
        bs = z.shape[0]
        z = self.lin_deco(z).view(bs, -1, 8, 8)
        return self.conv_deco(z)

class MyUnet(nn.Module):
    def __init__(self, encoder_blocks, decoder_blocks, hue_decoder):
        super().__init__()
        self.encoder_blocks = nn.ModuleList(encoder_blocks)
        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        self.hue_decoder = hue_decoder
    
    def forward(self, x, z):
        out = [x]
        
        for e in self.encoder_blocks:
            out.append(e(out[-1]))
        
        hue = self.hue_decoder(z)
        hue = F.interpolate(fv.TensorImage(hue), size = out[-1].shape[2:], mode='bilinear', align_corners=False)
        
        u = torch.cat((out[-1], hue),dim=1)
        x = self.decoder_blocks[0](u)
        
        out = reversed(out[:-1])
        for o,d in zip(out, self.decoder_blocks[1:]):
            u = torch.cat((x,o),dim=1)
            x = d(u)
        self.cm__hd = hue[:,:4]
        return x


def hue_dropout(x, p=(0.0,0.5)):
    p = torch.rand(x.shape[0], device=x.device)*(p[1]-p[0]) + p[0]
    
    mask = (p[:,None,None,None] < torch.rand_like(x[:,:1])).float()
    white = torch.tensor([1.0, 1.0, 1.0], device=x.device)
    return x*mask + white[None,:,None,None]*(1-mask)

class ZEncoder(nn.Module):
    def __init__(self,m):
        super().__init__()
        self.m = m
        
    def forward(self, x):
        x = F.interpolate(x, (96,96), mode='bilinear', align_corners=False)
        
        if self.training:
            x = hue_dropout(x)
        
        x = PositionalInfo()(x)
        return self.m(x)


class SuperModel(nn.Module):
    def __init__(self, unet, z_encoder):
        super().__init__()
        self.unet = unet
        self.z_encoder = z_encoder
    
    def forward(self, la, h):
        la = la[:, 0:1]
        z = self.z_encoder(h)
        return self.unet(la, z)

    def loss_func(self, yp, y):
        ye = self.unet.cm__hd
        yd = F.interpolate(y, size=ye.shape[2:], mode='bilinear', align_corners=False)
        yd = rgb2hhsv(yd)
        return F.mse_loss(yp, rgb2hhsv(y)) + F.mse_loss(ye, yd)
    
def get_model(z_features):
    
    lin_deco = nn.Sequential(*abl(z_features,256))
    conv_deco = nn.Sequential(ResBlock(4,16), ResBlock(16), ResBlock(16,32), ResBlock(32), ResBlock(32, 32))
    hue_deco = HueDecoder(lin_deco, conv_deco)
    
    encoder_blocks = [nn.Sequential(ResBlock(1, 32,s=2),ResBlock(32),ResBlock(32),ResBlock(32),
                                ResBlock(32, 64,s=2),ResBlock(64),ResBlock(64),ResBlock(64)),
                  nn.Sequential(ResBlock(64),ResBlock(64),ResBlock(64,128,s=2),
                                ResBlock(128),ResBlock(128),ResBlock(128),
                                ResBlock(128, 256, s=2), ResBlock(256), ResBlock(256))
                 ]

    decoder_blocks = [nn.Sequential(ResBlock(256+32,256),ResBlock(256),ResBlock(256),fv.PixelShuffle_ICNR(256,128),
                                    ResBlock(128),ResBlock(128),ResBlock(128),fv.PixelShuffle_ICNR(128,64),ResBlock(64)),
                      nn.Sequential(ResBlock(64+64),ResBlock(64+64,128),
                                    fv.PixelShuffle_ICNR(128,64),
                                    ResBlock(64),ResBlock(64),ResBlock(64),
                                    ResBlock(64),fv.PixelShuffle_ICNR(64,32),ResBlock(32)),
                      nn.Sequential(ResBlock(32+1,32),ResBlock(32),ResBlock(32),ResBlock(32,16),
                                    ResBlock(16),ResBlock(16),ResBlock(16),ResBlock(16,4),ResBlock(4))
                     ]

    unet = MyUnet(encoder_blocks, decoder_blocks, hue_deco)
    
    z_enc = ZEncoder(nn.Sequential(ResBlock(5,8,bottle=4),ResBlock(8),ResBlock(8,16,s=2),
                               ResBlock(16),ResBlock(16,32),ResBlock(32),ResBlock(32),
                               ResBlock(32),ResBlock(32),ResBlock(32,64,s=2),
                               ResBlock(64),ResBlock(64),ResBlock(64,128,s=2),
                               ResBlock(128),ResBlock(128),ResBlock(128,s=2),
                               nn.AdaptiveAvgPool2d(1),fv.Flatten(),
                               *abl(128,z_features),nn.Tanh()
                              ))
    return SuperModel(unet,z_enc)