import numpy as np


def Get_Uniform_RmveList(net_shape, pruning_ratio):

    '''
    Usage:
        To get the uniform remove list of the whole neural net based on a certain ratio
    Args:
        net_shape:     (list) the shape of the unpruned network
        pruning_ratio: (float) the ratio of channels to be removed
    '''

    rmve_list = (np.array(net_shape) * pruning_ratio).astype(int)
    return rmve_list


def Get_Default_Mask_From_Shape(net_shape):
    '''
    Usage:
        Obtain the all [True] default mask list for a given net shape

    Args:
        net_shape: (list) of number of channels in the layer
    '''
    net_default_mask = [np.array([True] * layer_shape) for layer_shape in net_shape]
    return net_default_mask

def Generate_Prune_Mask_List(Net_Score_List, net_shape, rmve_list, info_print = False):
    '''
    Usage:
        Get prune_mask_list by the channel score list and the removal list
    Args:
        Net_Score_List: (list) of layer (list) of scores of every channel
        net_shape:      (list) of of number of channels in the layer
        rmve_list:      (list) containing number of removed channels of every layer
        info_print:     (bool) whether to print the information or not
    '''

    net_mask_list = Get_Default_Mask_From_Shape(net_shape)
    print('\n' + '-----------------------------Actual Pruning Happens-----------------------------')

    for lay_k in range(len(net_shape)):

        layer_mask = net_mask_list[lay_k]
        layer_rmv = rmve_list[lay_k]
        layer_score_list = Net_Score_List[lay_k]
        assert len(layer_mask) == len(layer_score_list)

        if info_print:
            print('\n' + 'Layer ID: ' + str(lay_k))
            print('Layer Remove: ' + str(layer_rmv))

        # Pruning maps
        if (sum(layer_mask) > layer_rmv and layer_rmv > 0):
            rmv_node = np.argsort(layer_score_list)[:layer_rmv]
            layer_mask[rmv_node] = False

            print('We have masked out  #' + str(rmv_node) + ' in layer ' + str(lay_k) + '. It will have '
            + str(sum(layer_mask)) +' maps.')

    return net_mask_list