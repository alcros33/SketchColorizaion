# based on https://github.com/bobbens/sketch_simplification
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

BASE_DIR = Path(__file__).absolute().parent

simplify_model = nn.Sequential( # Sequential,
	nn.Conv2d(1,48,(5, 5),(2, 2),(2, 2)),
	nn.ReLU(),
	nn.Conv2d(48,128,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(128,128,(3, 3),(2, 2),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(128,256,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(256,256,(3, 3),(2, 2),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(256,512,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(512,1024,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(1024,512,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(512,256,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.ConvTranspose2d(256,256,(4, 4),(2, 2),(1, 1),(0, 0)),
	nn.ReLU(),
	nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(256,128,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.ConvTranspose2d(128,128,(4, 4),(2, 2),(1, 1),(0, 0)),
	nn.ReLU(),
	nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(128,48,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.ConvTranspose2d(48,48,(4, 4),(2, 2),(1, 1),(0, 0)),
	nn.ReLU(),
	nn.Conv2d(48,24,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(24,1,(3, 3),(1, 1),(1, 1)),
	nn.Sigmoid(),
)

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return list(zip_longest(*args, fillvalue=fillvalue))

def getTensorFromName(fname, immean =0.9664114577640158 , imstd=0.0858381272736797):
    data  = Image.open(fname).convert('L')
    return ((transforms.ToTensor()(data)-immean)/imstd)

class Normalizer(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean, self.std = torch.tensor(mean)[None,:,None,None], torch.tensor(std)[None,:,None,None]
        
    def forward(self, x):
        mean = self.mean.to(x)
        std = self.std.to(x)
        return (x - mean)/std

class SimplifyModel(nn.Module):
    def __init__(self, mean=0.9664114577640158 , std=0.0858381272736797):
        super().__init__()
        self.norm = Normalizer([mean], [std])
        self.model = simplify_model
        self.model.load_state_dict(torch.load(BASE_DIR/"models"/"sketch_simplification.pth"))
        
    def forward(self, x):
        return self.model(self.norm(x))