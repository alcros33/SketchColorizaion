import torch.nn.functional as F
import torch.nn as nn
from fastai.callback.hook import hook_outputs

def gram_matrix(x):
    bs,c,h,w = x.shape
    x = x.view(bs, c, h*w)
    return (x @ x.transpose(1,2))/(c*h*w)

class PerceptualLoss(nn.Module):
    def __init__(self, model, layer_ids, content_weights=[1,3,10,15,20,15],  style_weights=[1,15,20,20,12,6], style_mult=1e3):
        super().__init__()
        model.eval()
        for p in model.parameters(): p.requires_grad_(False)
        
        self.model = model
        self.important_layers = [self.model[i] for i in layer_ids]
        self.hooks = hook_outputs(self.important_layers, detach=False)
        self.content_weights = content_weights
        self.style_weights = style_weights
        self.style_mult = style_mult

    def extract_features(self, x, clone=False):
        self.model(x)
        features = list(self.hooks.stored)
        
        if clone:
            features = [f.clone() for f in features]
        
        return features
    
    def content_loss(self, A, B):
        return sum([F.smooth_l1_loss(a, b)*w for a, b, w in zip(A, B, self.content_weights)])
    
    def style_loss(self, A, B):
        return sum([F.smooth_l1_loss(gram_matrix(a), gram_matrix(b))*w for a, b, w in zip(A, B, self.style_weights)])
    
    def forward(self, yp, y):
        yp_features = self.extract_features(yp)
        y_features = self.extract_features(y, clone=True)
        
        self.CL = self.content_loss(yp_features, y_features)
        self.SL = self.style_mult*self.style_loss(yp_features, y_features)

        return self.CL + self.SL
    
    def __del__(self): 
        self.hooks.remove() # necesario para que deje de guardar las cosas
