from pathlib import Path
import argparse, sys
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import fastai.vision.all as fv

def xyv_perceptual_loss(yp, y):
    return 0

def get_y(fname: Path):
    return fname

def get_hint(name: Path):
    global hintname
    return fv.PILImage.create(hintname)

def path_file(p):
    path = Path(p)
    if path.is_file():
        return path
    else:
        raise argparse.ArgumentTypeError(f"{path} is not a file")

BASE_DIR = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description='Colorize linearts')
parser.add_argument('lineart', type=path_file)
parser.add_argument('colorhint', type=path_file)
parser.add_argument("--learner", type=path_file, default=(BASE_DIR/"models")/"colorizer.pkl")
parser.add_argument("--savename", default=None)
        
def main():
    global hintname
    args = parser.parse_args()
    
    learn = fv.load_learner(args.learner)
    
    input_file = Path(args.lineart)
    hintname = args.colorhint
    
    res = learn.predict(input_file)
    img = fv.PILImage.create(res[0])
    
    if args.savename is None:
        args.savename = input_file.with_name(input_file.stem+"_color").with_suffix(input_file.suffix)
    img.save(args.savename)
    

if __name__ == "__main__":
    main()