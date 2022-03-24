import torch
import torch.nn as nn
import torch.nn.functional as F
import fastai.vision.all as fv

def avg_conv_weight(ni,k,s,device='cpu'):
    T = torch.zeros(ni,ni,k,k,device=device)
    I = torch.arange(ni,device=device)
    a = (k-s)//2
    b = a+s
    T[I,I,a:b,a:b] = 1/s**2
    return T

def init_id_conv(conv, wd=1., nc = None):
    conv.init_id = True
    conv.wd = wd
    k = conv.kernel_size[0]
    s = conv.stride[0]
    g = conv.groups
    assert(g==1 and "Not implemented correctly otherwise, I think")
    with torch.no_grad():
        T = conv.weight
        no,ni,k,k = T.shape
        if nc is None:
            nc = min(no,ni*g)
            
        if conv.bias is not None: 
            nn.init.constant_(conv.bias[:nc], 0.)
            
        assert(0 < nc <= min(no,ni*g))
        
        for i in range(g):
            c = min(ni,nc,no//g)
            A = avg_conv_weight(c,k,s,T.device)
            #print(f"group {i}: A.shape = {A.shape}, c={c}, nc={nc}, ni={ni}, no={no}")
            if (c < ni):
                Z = torch.zeros((c,ni-c,k,k),device=T.device)
                A = torch.cat((A,Z),dim=1)
            start = i*no//g
            end = start + c
            T[start:end] = A
            nc -= c
            if nc == 0:
                break

def init_bn_to_0(bn, wd=1.):
    nn.init.constant_(bn.weight, 0.)
    bn.init_zero = True
    bn.wd = wd

def _decay_to_0(T,pre_div=2.):
    t = T/pre_div
    return (t*t).mean()
    
def _decay_to_identity(w,s):
    no,ni,k,k = w.shape
    #assert(no==ni)
    n = min(ni,no)
    I = avg_conv_weight(n,k,s,device=w.device)
    
    return F.mse_loss(w[:n,:n],I)

def bn_loss(bn,surge_protection):
    if not bn.affine: return 0.
    w = bn.weight.float()
    b = bn.bias.float()
    
    wd = bn.wd if hasattr(bn,'wd') else surge_protection
    init_zero = (hasattr(bn,'init_zero') and bn.init_zero)
    pre_div = 1. if init_zero else 2.
    
    t = torch.zeros_like if init_zero else torch.ones_like
    return F.mse_loss(w,t(w))*wd + _decay_to_0(b,pre_div=pre_div)*surge_protection

def _get_conv_weight_loss(c,surge_protection):
    wd = c.wd if hasattr(c,'wd') else 1.
    w = c.weight.float()
    
    no,ni,k,k = w.shape
    

    init_id = (hasattr(c,'init_id') and c.init_id)
    n_id = ni if init_id else 0
    pre_div = 1. if init_id else 2.
    
    loss = 0.
    
    if n_id < no and surge_protection > 0:
        loss += _decay_to_0(w[n_id:],pre_div=pre_div)*surge_protection 
    if n_id > 0 and wd > 0:
        s = c.stride[0]
        loss += _decay_to_identity(w[:n_id],s)*wd
    return loss

def conv_loss(c,surge_protection):
    loss = _get_conv_weight_loss(c,surge_protection)
    
    if c.bias is not None and surge_protection > 0:
        b = c.bias.float()
        loss += surge_protection*(_decay_to_0(b))

    return loss

def model_loss(model,surge_protection=0.1):
    loss = 0.
    modules = fv.flatten_model(model)
    for m in modules:
        if isinstance(m, nn.Conv2d):
            loss += conv_loss(m,surge_protection)
        elif isinstance(m, nn.BatchNorm2d):
            loss += bn_loss(m,surge_protection)
    return loss

class WeightDecaySmart(fv.Callback):
    run_after,run_valid = fv.TrainEvalCallback,False
    
    def __init__(self,pct=0.4,start=0.005,middle=0.01,end=0.0001):
        self.sched = fv.combined_cos(pct,start,middle,end)
        self.swd = self.sched(0.)
    
    def after_loss(self):
        self.swd = self.sched(self.pct_train)
        self.learn.loss_grad += self.swd*model_loss(self.model)
