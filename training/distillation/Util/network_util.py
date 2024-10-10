from torchvision import utils
import torch
from torch import nn
from PIL import Image

def Convert_Tensor_To_Image(img_tensor):
    '''
    Usage:
        Convert a torch.Tensor output from the StyleGAN2 to a PIL image
    '''
    grid = utils.make_grid(img_tensor, nrow =1, padding=2, pad_value=0,
                           normalize=True, range = (-1,1), scale_each=False)
    # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    return im


def Get_Network_Shape(generator):
    '''
    Usage:
        Return the shape of the network (number of channels in each layer) in a list

    Args:
        model_dict: (dict) of a StyleGAN2 generator
    '''
    conv_key_list = [n for n, p in generator.named_parameters()
                 if ("conv" in n and "weight" in n and not ("affine" in n)
                     or n == f"synthesis.b4.conv1.weight")]

    num_channels = [generator.state_dict()[key].shape[1] for key in conv_key_list]
    num_channels.append(generator.state_dict()[conv_key_list[-1]].shape[0])

    return num_channels


def Get_Conv_Kernel_Key(model_dict):
    '''
    Usage:
        Return a list of keys of the convolutional weights in main feedforwarding flow

    Args:
        model_dict: (dict) of a StyleGAN2 generator.
    '''
    CONV1_KEY = 'synthesis.b4.conv1.weight'

    conv_key_list = [CONV1_KEY]
    for key in model_dict.keys():
        if ('conv' in key) and ('weight' in key):
            conv_key_list.append(key)

    return conv_key_list