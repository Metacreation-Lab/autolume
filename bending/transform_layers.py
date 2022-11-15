import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import utils
import torchvision
import random
import os
import kornia

class Sobel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, params):
        if params[0]:
                # raise ValueError
            tf = kornia.filters.sobel(x)
            return x * tf
        return x

class Canny(nn.Module): #find a way to make faster
    def __init__(self):
        super().__init__()
    
    def forward(self, x, params, indices):
        print(x.shape)
        if params[0]:
                # raise ValueError
            _, tf = kornia.filters.canny(x.permute(1,0,2,3)[indices])
            return x[:,indices] * tf.permute(1,0,2,3)
        return x

class Erode(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, params, indices):
        if(not isinstance(params[0], int) or params[0] < 0):
            print('Erosion parameter must be a positive integer')
            # raise ValueError
        print(params)
        x[:, indices] = kornia.morphology.erosion(x[:, indices], torch.ones((params[0],params[0]), device=x.device).to(x.dtype), engine="convolution")
        return x

class Dilate(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, params, indices):
        if(not isinstance(params[0], int) or params[0] < 0):
            print('Dilation parameter must be a positive integer')
            # raise ValueError
        x[:, indices] = kornia.morphology.dilation(x[:, indices], torch.ones((params[0],params[0]), device=x.device).to(x.dtype), engine="convolution")
        return x

class Translate(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, params, indices):
        print(x.shape)
        x[:,indices] = kornia.geometry.transform.translate(x[:, indices],
                                                           torch.tensor([params], device=x.device).to(x.dtype))
        return x

class Resize(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, params, indices):
        tf = kornia.geometry.transform.resize(x[:, indices], torch.tensor([params], device=x.device).to(x.dtype))
        x[:,indices] = torchvision.transforms.CenterCrop(x.shape[-2:])(tf)
        return x

class Scale(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, params, indices):
        if params[0] == 0:
            p = 0.0000000001
        else:
            p = float(params[0])
        tf = kornia.geometry.transform.scale(x[:, indices],torch.tensor(p, device=x.device, dtype=x.dtype))
        x[:,indices] = torchvision.transforms.CenterCrop(x.shape[-2:])(tf)
        return x

class Rotate(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, params, indices):
        x[:, indices] = kornia.geometry.transform.rotate(x[:,indices],torch.tensor(float(params[0]), device=x.device, dtype=x.dtype))
        return x

class FlipHorizontal(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, params, indices):
        if params[0]:
            x [:, indices] = x[:, indices].flip(2)
        return x

class FlipVertical(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, params, indices):
        if params[0]:
            x[:, indices] = x[:, indices].flip(3)
        return x

class Invert(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, params, indices):
        if params[0]:
            ones = torch.ones(x[:, indices].shape, dtype=x.dtype, layout=x.layout, device=x.device)
            x[:, indices] = ones - x[:, indices]
            return x
        return x

class BinaryThreshold(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, params, indices):
        if(not isinstance(params[0], float) or params[0] < -1 or params[0] > 1):
            print('Binary threshold parameter should be a float between -1 and 1.')
            # raise ValueError
        x[:, indices] = (x[:, indices] > params[0]).to(x.dtype)
        return x

class ScalarMultiply(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, params, indices):
        if(not isinstance(params[0], float)):
            print('Scalar multiply parameter should be a float')
            # raise ValueError
        x[:, indices] *= params[0]
        return x

class Ablate(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, params, indices):
        if params[0]:
            x[:, indices] *= 0
        return x

class ManipulationLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # layers
        self.erode = Erode()
        self.dilate = Dilate()
        self.translate = Translate()
        self.scale = Scale()
        self.rotate = Rotate()
        self.resize = Resize()
        self.flip_h = FlipHorizontal()
        self.flip_v = FlipVertical()
        self.invert = Invert()
        self.binary_thresh = BinaryThreshold()
        self.scalar_multiply = ScalarMultiply()
        self.ablate = Ablate()
        self.sobel = Sobel()
        self.canny = Canny()
        
        self.layer_options = {
            "sobel": self.sobel,
            "canny": self.canny,
            "erode" : self.erode,
            "dilate": self.dilate,
            "translate": self.translate,
            "scale": self.scale,
            "rotate": self.rotate,
            "resize": self.resize,
            "flip-h": self.flip_h,
            "flip-v": self.flip_v,
            "invert": self.invert,
            "binary-thresh": self.binary_thresh,
            "scalar-multiply": self.scalar_multiply,
            "ablate": self.ablate
        }

    def forward(self, input, transform_dict):
        out = input
        if transform_dict.get('indices', [0]):
            out = self.layer_options[transform_dict['transformID']](input, transform_dict['params'],
                                                                    transform_dict.get('indices', [0]))
        return out
    