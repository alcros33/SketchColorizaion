from math import pi, tau
import torch

def rgb2hsv(rgb, eps = 1e-7):
    hsv = torch.zeros_like(rgb)
    
    MAX, iM = rgb.max(1)
    MIN, im = rgb.min(1)
    D = MAX - MIN

    hsv[:,2] = MAX # value
    hsv[:,1] = D/(MAX+eps) # saturation
    
    D += eps
    
    RGB = rgb/D[:,None]
    R,G,B = RGB[:,0],RGB[:,1],RGB[:,2]
    
    isR, isG, isB = (iM==0).float(), (iM==1).float(), (iM==2).float()
    hsv[:,0] = (((G-B))*isR + (2. + (B-R))*isG +  (4. + (R-G))*isB)/6.
    hsv[:,0][hsv[:,0] < 0.] += 1.

    return hsv

def hsv2rgb(hsv,eps=1e-7):
    H,S,V = hsv[:,0:1]*6, hsv[:,1:2], hsv[:,2:3]
    C = V*S
    X = C*(1 - torch.abs( H%2 - 1 ))
    m = V - C

    Z = torch.zeros_like(C)

    CXZ = torch.cat((C,X,Z),dim=1)
    XCZ = torch.cat((X,C,Z),dim=1)
    ZCX = torch.cat((Z,C,X),dim=1)
    ZXC = torch.cat((Z,X,C),dim=1)
    XZC = torch.cat((X,Z,C),dim=1)
    CZX = torch.cat((C,Z,X),dim=1)

    P = [CXZ,XCZ,ZCX,ZXC,XZC,CZX]

    I = H.long()

    #I[I==6] = 0
    
    RGB = sum([(I==i)*p for i,p in enumerate(P)])

    return m + RGB

# HHSV

def hsv2hhsv(hsv):
    h = hsv[:,0:1]*tau-pi
    return torch.cat((torch.cos(h), torch.sin(h), hsv[:,1:]),dim=1)

def hhsv2hsv(hhsv):
    hue = (torch.atan2(hhsv[:,1],hhsv[:,0]) + pi)/tau
    sv = hhsv[:,2:]
    return torch.cat((hue[:,None],sv),dim=1).clamp(0,1)

def hhsv2rgb(o): return hsv2rgb(hhsv2hsv(o))
def rgb2hhsv(o): return hsv2hhsv(rgb2hsv(o))

# XYV

def xyv2hhsv(x):
    s = torch.sqrt(x[:,0:1]*x[:,0:1] + x[:,1:2]*x[:,1:2])
    return torch.cat([x[:,:2]/(s+1e-6),s,x[:,2:]],dim=1)

def hhsv2xyv(x):
    s = x[:,2:3]
    return torch.cat([x[:,:2]*s, x[:,3:]],dim=1)

def rgb2xyv(x):
    hsv = rgb2hsv(x)
    h = hsv[:,0:1]*tau-pi
    s = hsv[:,1:2]
    return torch.cat((s*torch.cos(h), s*torch.sin(h), hsv[:,2:]),dim=1)

def xyv2rgb(x):
    return hhsv2rgb(xyv2hhsv(x))

