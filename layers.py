import random, inspect
from math import prod,sqrt, log, exp
import fastai.vision.all as fv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from weightdecay import *

def delegates(to=None, keep=False):
    "Decorator: replace `**kwargs` in signature with params from `to`"
    def _f(f):
        if to is None: to_f,from_f = f.__base__.__init__,f.__init__
        else:          to_f,from_f = to,f
        sig = inspect.signature(from_f)
        sigd = dict(sig.parameters)
        k = sigd.pop('kwargs')
        s2 = {k:v for k,v in inspect.signature(to_f).parameters.items()
              if v.default != inspect.Parameter.empty and k not in sigd}
        sigd.update(s2)
        if keep: sigd['kwargs'] = k
        from_f.__signature__ = sig.replace(parameters=sigd.values())
        return f
    return _f


relu = nn.ReLU(inplace=False)
relu_i = nn.ReLU(inplace=True)

leaky = nn.LeakyReLU(1/16, inplace=False)
leaky_i = nn.LeakyReLU(1/16, inplace=True)

celu = nn.CELU(inplace=False)
celu_i = nn.CELU(inplace=True)

tanh = nn.Tanh()

default_act = relu
default_act_i = relu_i

def num_params(model):
    return sum([prod(p.shape) for p in model.parameters()])


def replace_module_by_other(model,module,other):
    for child_name, child in model.named_children():
        if isinstance(child, module):
            setattr(model, child_name, other)
        else:
            replace_module_by_other(child,module,other)

def identity(x):
    return x

def conv2d(ni,no,k=3,s=1,pad="same",g=1,init='none',bias=True,nc=None):
    #assert(k%s == 0)
    if pad=="same": pad = (k-1)//2

    conv = nn.Conv2d(ni,no,kernel_size=k,stride=s,padding=pad,groups=g,bias=bias)

    if bias:
        nn.init.constant_(conv.bias,0.)

    if not fv.is_listy(init): init = [init]
    
    if 'linear' in init:
        nn.init.kaiming_normal_(conv.weight, nonlinearity=init)
    else:
        nn.init.kaiming_normal_(conv.weight, nonlinearity='relu')

    if 'id' in init or 'avg' in init: init_id_conv(conv,nc=nc)

    return conv


@delegates(conv2d)
def acb(ni, no, bn=True, activation=True, act_fn=default_act, bn_init_zero=False, p = 0., init='relu', **kwargs):
    layers = []
    
    if activation: layers += [act_fn]
    
    layers += [conv2d(ni, no, init=init, bias=(not bn), **kwargs)]
    
    if bn:
        layers += [nn.BatchNorm2d(no)]
        if bn_init_zero: init_bn_to_0(layers[-1])

    if p > 0: layers += [nn.Dropout2d(p)]

    return layers


@delegates(conv2d)
def abc(ni, no, bn=True, activation=True, act_fn=default_act, bn_init_zero=False, p = 0., **kwargs):
    layers = []
    
    if activation: layers += [act_fn]
    if p > 0: layers += [nn.Dropout2d(p)]

    if bn:
        layers += [nn.BatchNorm2d(ni)]
        if bn_init_zero: init_bn_to_0(layers[-1])

    layers += [conv2d(ni, no, bias=(not bn), **kwargs)]

    return layers


@delegates(conv2d)
def cab(ni, no, bn=True, activation=True, act_fn=default_act, bn_init_zero=False, p = 0., init='relu', **kwargs):
    layers = []
    
    layers += [conv2d(ni, no, init=init, **kwargs)]
    
    if activation: layers += [act_fn]
    if p > 0: layers += [nn.Dropout2d(p)]

    if bn:
        layers += [nn.BatchNorm2d(no)]
        if bn_init_zero: init_bn_to_0(layers[-1])

    return layers


@delegates(conv2d)
def cba(ni, no, bn=True, activation=True, act_fn=default_act, bn_init_zero=False, p = 0., init='relu', **kwargs):
    #if init == 'avg' or init == 'identity':
        #init = [init]
    layers = []
    
    layers += [conv2d(ni, no, init=init, bias=(not bn), **kwargs)]
    
    if bn:
        layers += [nn.BatchNorm2d(no)]
        if bn_init_zero: init_bn_to_0(layers[-1])

    if activation: layers += [act_fn]
    if p > 0: layers += [nn.Dropout2d(p)]

    return layers

@delegates(acb)
def acb_block(ni,no,**kwargs):
    return nn.Sequential(*acb(ni,no,**kwargs))

@delegates(abc)
def abc_block(ni,no,**kwargs):
    return nn.Sequential(*abc(ni,no,**kwargs))

@delegates(cab)
def cab_block(ni,no,**kwargs):
    return nn.Sequential(*cab(ni,no,**kwargs))

@delegates(cba)
def cba_block(ni,no,**kwargs):
    return nn.Sequential(*cba(ni,no,**kwargs))

##############################################################

def alb(ni, no, bn=True, activation=True, act_fn=leaky_i, bn_init_zero=False, p = 0.):
    layers = []
    
    if p > 0: layers += [nn.Dropout(p)]
    if activation: layers += [act_fn]
    
    layers += [nn.Linear(ni, no, bias=(not bn))]
    
    if bn:
        layers += [nn.BatchNorm1d(ni)]
        if bn_init_zero:
            init_bn_to_0(layers[-1])
    return layers

def lab(ni, no, bn=True, activation=True, act_fn=leaky_i, bn_init_zero=False, p = 0.,init='none'):
    layers = []
    
    layers += [nn.Linear(ni, no)]
    if 'id' in init or 'avg' in init: init_id(layers[-1])
        
    if activation: layers += [act_fn]
    if p > 0: layers += [nn.Dropout(p)]

    if bn:
        layers += [nn.BatchNorm1d(no)]
        
        if bn_init_zero:
            init_bn_to_0(layers[-1])  
    return layers

def lba(ni, no, bn=True, activation=True, act_fn=leaky_i, bn_init_zero=False, p = 0.):
    layers = []
    
    layers += [nn.Linear(ni, no, bias=(not bn))]

    if bn:
        layers += [nn.BatchNorm1d(no)]
        
        if bn_init_zero:
            init_bn_to_0(layers[-1])  
    
    if activation: layers += [act_fn]
    if p > 0: layers += [nn.Dropout(p)]
    
    return layers

def abl(ni, no, bn=True, activation=True, act_fn=leaky, bn_init_zero=False, p = 0.):
    layers = []
    
    if activation: layers += [act_fn]
    if p > 0: layers += [nn.Dropout(p)]

    if bn:
        layers += [nn.BatchNorm1d(ni)]
        
        if bn_init_zero:
            init_bn_to_0(layers[-1])  
            
    layers += [nn.Linear(ni, no, bias=(not bn))]
    
    return layers

@delegates(alb)
def alb_block(ni,no,**kwargs):
    return nn.Sequential(*alb(ni,no,**kwargs))

class PositionalInfo(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        bs,c,h,w = x.shape
        H = torch.arange(0,1,1/h)
        W = torch.arange(0,1,1/w)
        H = H[None,None,:,None].expand(bs,1,h,w).to(x)
        W = W[None,None,None,:].expand(bs,1,h,w).to(x)
        return torch.cat([H,W,x],dim=1)

def _get_bn_weight(A):
    if not isinstance(A, nn.Sequential):
        return 1
    if isinstance(A[-1], nn.BatchNorm2d):
        return A[-1].weight[None,:,None,None]
    return 1

class SmartAdd(nn.Module):
    def __init__(self,A,B):
        super().__init__()
        self.pathA = A
        self.pathB = B
    
    def forward(self, x):
        A, B = self.pathA, self.pathB
        #return A(ax) + B(ax)
        
        a = _get_bn_weight(A)
        b = _get_bn_weight(B)
        
        divider = torch.sqrt(a*a + b*b) + 1e-6
        divider = divider.to(x)
        
        return (A(x) + B(x))/divider
    
class Concat(nn.Module):
    def __init__(self,A,B):
        super().__init__()
        self.pathA = A
        self.pathB = B
        
    def forward(self, x):
        A, B = self.pathA, self.pathB
        
        return torch.cat((A(x),B(x)),dim=1)

class Cascade(nn.Module):
    def __init__(self,nf,steps,act_fn=default_act):
        super().__init__()
        assert(nf%steps == 0)
        t = nf//steps
        
        self.t = t
        
        modules = [acb_block(t,t,act_fn=act_fn) for _ in range(steps-1)]
        self.convs = nn.ModuleList(modules)
    
    def forward(self, x):
        t, convs = self.t, self.convs
        
        p = torch.split(x,t,dim=1)
        out = [p[0]]
        prev = 0.
        for conv, z in zip(convs, p[1:]):
            prev = conv(prev+z)
            out.append(prev)
        return torch.cat(out,dim=1)

class Stair(nn.Module):
    """groups should be an integer, or 'full' or 'mid' or a list of no//steps - 1 integers"""
    def __init__(self, ni, no=None, steps=4, k=3, s=1, groups = 1, act_fn=default_act):
        super().__init__()
        if no is None: no = ni
        
        if fv.is_listy(groups): steps = len(groups)+1
        p = steps
        
        assert(ni%p == 0 and no%p==0)
        t = no//p
        
        if type(groups) == int: groups = [groups]*(p-1)
        
        gg = groups
        if gg in ['full', 'mid']: groups = [p-i for i in range(1,p)]
        if gg == 'mid':
            groups = [(g if g%2==1 else g//2) for g in groups]
                
        assert(len(groups) == p-1)
        #print("groups = ", groups)
        self.t = t
        
        modules = [conv2d(ni,no,k=k,s=s)]
        modules += [abc_block((p-i)*t,(p-i)*t,act_fn=act_fn,g=g,init='avg') for i,g in zip(range(1,p),groups)]
        self.convs = nn.ModuleList(modules)
    
    def forward(self, x):
        t, convs = self.t, self.convs
        
        out = []
        
        for c in convs:
            #print(f"{i}: {c[1].in_channels}->{c[1].out_channels}, and x.shape={x.shape}")
            x = c(x)
            out.append(x[:,:t])
            x = x[:,t:]
            
        return torch.cat(out,dim=1)

class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1,1,1)
        else:
            x = torch.sigmoid(x)
        return x

class Splat(nn.Module):
    def __init__(self, ni, no, radix=2, g=1, reduction=4):
        super().__init__()
        self.no = no
        self.cab = cab_block(ni,no*radix,g=g*radix)
        
        nm = max((ni*radix)//reduction,32)
        
        self.attention = nn.Sequential(SmartPoolNF(),
                                       conv2d(no,nm,k=1,g=g),
                                       *abc(nm,no*radix,k=1,g=g),
                                       rSoftMax(radix,g))
        
    def forward(self, x):
        x = self.cab(x)
        no = self.no
        split = torch.split(x, no, dim=1)
        attention = self.attention(sum(split))
        attentions = torch.split(attention, no, dim=1)
        return sum([a*s for a,s in zip(attentions, split)])

class PoolSplat(nn.Module):
    def __init__(self, nf, bottle=None, g=4):
        super().__init__()
        self.g = g
        if bottle is None: bottle = max(nf//2,32)
        
        nm = bottle*g
        self.attention = nn.Sequential(abc_block(nf,nm,k=3,g=g),
                                       nn.MaxPool2d(2),
                                       nn.BatchNorm2d(nm),
                                       SmartPoolNF(),
                                       *abc(nm,nm,k=1,g=g),
                                       *abc(nm,nf,k=1,g=g))
    def split_softmax(self, x, g):
        bs,c = x.shape[:2]
        x_mat = x.view(bs,g,-1)
        return torch.softmax(x_mat,dim=2).view(bs,c,1,1)*c/g
    
    def forward(self, x):
        attention = self.attention(x)
        return x*self.split_softmax(attention,self.g)

def residual_builder(res_func, ni, no=None, s=1, use_pool=False, pre_activate=True, pre_act=celu, post_bn=True, **kwargs):
    if no is None: no = ni

    k = 3 if s == 1 else 4
        
    pool = identity
    if use_pool or s == 2 or no != ni:
        pool = conv2d(ni,no,k=k,s=s,init='avg')
        pre_activate=True
        post_bn=True

    layers = []

    if pre_activate:
        layers += [pre_act]
    if post_bn:
        layers += [nn.BatchNorm2d(ni)]

    residual = res_func(ni=ni,no=no,k=k,s=s,**kwargs)
    
    if not pre_activate:
        residual = [pre_act,*residual]

    residual = nn.Sequential(*residual)
    rblock = SmartAdd(pool,residual)
    layers += [rblock]
    
    return nn.Sequential(*layers)
    
@delegates(residual_builder)
def ResBlock(ni, no=None, bottle=None, g=1, **kwargs):
    def _normal_residual(ni,no,bottle,k,s,g):
        if bottle is None: bottle = max(ni//2,2)
        return nn.Sequential(*cba(ni,bottle,g=g,activation=False),
                             *acb(bottle,no,k=k,s=s,g=g,bn_init_zero=True))
    
    return residual_builder(_normal_residual, ni, no, bottle=bottle, g=g, **kwargs)

@delegates(residual_builder)
def ResStairBlock(ni, no=None, bottle=None, steps=4, groups = 1, **kwargs):
    def _stair_residual(ni,no,bottle,k,s,steps,groups):
        if bottle is None: bottle = ni//2
        return nn.Sequential(Stair(ni, bottle, k=k, s=s, steps=steps, groups=groups),
                             *acb(bottle, no, k=1, s=1, bn_init_zero=True))
    
    return residual_builder(_stair_residual, ni, no, bottle=bottle, steps=steps, groups=groups, **kwargs)

@delegates(residual_builder)
def Res2Block(ni, no=None, bottle=None, g=1, **kwargs):
    def _cascade_residual(ni,no,bottle,k,s,g):
        if bottle is None: bottle = ni//2
        return nn.Sequential(*acb(ni, bottle, k=k, s=s, activation=False),
                             Cascade(bottle, steps=g),
                             *acb(bottle, no, k=1, s=1, bn_init_zero=True))
    
    return residual_builder(_cascade_residual, ni, no, bottle=bottle, g=g, **kwargs)

@delegates(residual_builder)
def ResSplatBlock(ni, no=None, bottle=None, radix=2, g=1, reduction=2, **kwargs):
    def _splat_residual(ni,no,bottle,k,s,g,radix,reduction):
        if bottle is None: bottle = ni
        return nn.Sequential(Splat(ni, bottle, radix=radix, g=g, reduction=reduction),
                             *acb(bottle,no,k=k,s=s,g=g,bn_init_zero=True))
    
    return residual_builder(_splat_residual, ni, no, bottle=bottle, g=g, radix=radix,reduction=reduction, **kwargs)

@delegates(residual_builder)
def ResPoolSplatBlock(ni, bottle=None, g=4, **kwargs):
    def _splat_residual(ni,no,k,s,bottle,g):
        return nn.Sequential(PoolSplat(ni, bottle=bottle, g=g, reduction=reduction),
                             *acb(ni, no, k=k, s=s, bn_init_zero=True))
    
    return residual_builder(_splat_residual, ni, no, bottle=bottle, g=g, **kwargs)

class RandomSizeCropAndResizeBatch(fv.RandTransform):
    "Picks a random scaled crop of an image and resize it to `size`"
    split_idx,order = None,30
    def __init__(self, sizes, min_scale=0.8, ratio=(3/4, 4/3), valid_scale=1., **kwargs):
        sizes = [(s,s) if type(s)==int else s for s in sizes]
        fv.store_attr()
        super().__init__(**kwargs)

    def before_call(self, b, split_idx):
        self.do = True
        
        if split_idx:
            self.size = self.sizes[-1]
            self.mode = 'bilinear'
            self.align_corners = True
        else:
            self.size = random.choice(self.sizes)
            self.mode = random.choice(['nearest', 'bilinear'])
            self.align_corners = None if self.mode == 'nearest' else random.choice([False,True])
            
            #print(f"size = {self.size}, mode = {self.mode}")
        
        
        h,w = fv.fastuple((b[0] if isinstance(b, tuple) else b).shape[-2:])
        for attempt in range(10): # this should always happen on the first try!
            if split_idx: break
            area = random.uniform(self.min_scale,1.) * w * h
            
            a,b = (log(r) for r in self.ratio)

            a = max(a,log(area/h**2))
            b = min(b,log(w**2/area))
            ratio = exp(random.uniform(a,b))
            nw = int(round(sqrt(area * ratio)))
            nh = int(round(sqrt(area / ratio)))
            if nw <= w and nh <= h:
                self.cp_size = (nh,nw)
                self.tl = random.randint(0,h - nh),random.randint(0,w-nw)
                return
        if   w/h < self.ratio[0]: self.cp_size = (int(w/self.ratio[0]), w)
        elif w/h > self.ratio[1]: self.cp_size = (h, int(h*self.ratio[1]))
        else:                     self.cp_size = (h, w)
        if split_idx: self.cp_size = (int(self.cp_size[0]*self.valid_scale), int(self.cp_size[1]*self.valid_scale))
        self.tl = ((h-self.cp_size[0])//2,(w-self.cp_size[1])//2)

    def encodes(self, x:fv.TensorImage):
        x = x[...,self.tl[0]:self.tl[0]+self.cp_size[0], self.tl[1]:self.tl[1]+self.cp_size[1]]
        return fv.TensorImage(x).affine_coord(sz=self.size, mode=self.mode, align_corners=self.align_corners)

def get_possible_sizes(min_x,max_x,min_y,max_y,stride=32,max_aspect=1.3):
    sizes = []
    for x in range(min_x,max_x+1,stride):
        for y in range(min_y,max_y+1,stride):
            if x/y <= max_aspect and y/x <= max_aspect:
                sizes.append((x,y))
    return sizes
    

class Normalize(nn.Module):
    def __init__(self, mean = fv.imagenet_stats[0], std = fv.imagenet_stats[1]):
        super().__init__()
        mean = torch.tensor(mean)
        std = torch.tensor(std)
        assert(len(mean.shape) == 1)
        assert(len(std.shape) == 1)
        self.mean = mean[None,:,None,None]
        self.std = std[None,:,None,None]

    def forward(self, x):
        m,s = self.mean.to(x),self.std.to(x)
        return (x-m)/s

def flatten(x):
    bs = x.shape[0]
    return x.reshape(bs,-1)

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(x):
        return flatten(x)

class SmartPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        b = sqrt(x.shape[2]*x.shape[3])/8 # Theoretically I should substract 1, but whatever. /8 to avoid fp16 problems
        return flatten(self.ap(x))*b
    
        #bs = a.shape[0]
        #s = torch.tensor(x.shape[2:]).to(x)[None,:].repeat(bs,1)
        #return torch.cat((s,a),dim=1)
class SmartPoolNF(nn.Module):
    def __init__(self):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool2d(1)
        #self.mp = nn.AdaptiveMaxPool2d(1)
        
    def forward(self, x):
        b = sqrt(x.shape[2]*x.shape[3])/8 # Theoretically I should substract 1, but whatever. /8 to avoid fp16 problems
        return self.ap(x)*b
        
class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):  
        return x*torch.tanh(F.softplus(x))
    
def add_module_to_seq_model(model, module, i="last"):
    layers = list(model)
    if i == "last":
        layers.append(module)
    else:
        layers = layers[:i] + [module] + layers[i:]
    return nn.Sequential(*layers)

class LossAdder(fv.Callback):
    def __init__(self, addToLoss):
        self.addToLoss = addToLoss
    def after_loss(self):
        yp,y,x,model = self.learn.pred, self.learn.y, self.learn.x, self.learn.model
        self.learn.loss_grad += self.addToLoss(yp,y,x,model)
        self.learn.loss = self.learn.loss_grad.clone()

def zero_loss(yp,y):
    return torch.zeros((1,),device=yp.device)

class Reshaper(nn.Module):
    def __init__(self, h, w):
        super().__init__()
        self.h, self.w = h,w
        
    def forward(self, x):
        bs,c_in = x.shape
        c_out = c_in//(self.h*self.w) # hopefully its a divisor?
        return x.reshape(bs, c_out, self.h, self.w)