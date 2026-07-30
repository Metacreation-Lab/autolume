"""Network shape helpers shared by the pkl loader and training code."""


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
