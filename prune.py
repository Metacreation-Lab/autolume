import copy
import os
import pickle
from datetime import datetime

import torch
import click
import time
import numpy as np
import torchvision.utils

import dnnlib
from torch_utils import legacy
from training.distillation.Util.content_aware_pruning import Get_Content_Aware_Pruning_Score
from training.distillation.Util.mask_util import Mask_the_Generator
from training.distillation.Util.network_util import Get_Network_Shape
from training.distillation.Util.pruning_util import Get_Uniform_RmveList, Generate_Prune_Mask_List


@click.command()
@click.argument('pkl', metavar='PATH', nargs=-1)
@click.option('--outdir', help='Root directory for run results (default: %(default)s)', default='results', show_default=True)
@click.option('--n_samples', type=int, default=400)
@click.option('--batch_size', type=int, default=10)
@click.option('--noise_prob', type=float, default=0.05)
@click.option('--remove_ratio', type=float, default=0.7)
@click.option('--custom', default=True)
@click.option('--info_print', default=False)
def clickmain(pkl, outdir, n_samples, batch_size, noise_prob, remove_ratio, custom, info_print):
    main(pkl, outdir, n_samples, batch_size, noise_prob, remove_ratio, custom, info_print)

def main(pkl, outdir, n_samples, batch_size, noise_prob, remove_ratio, custom=True, info_print=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("CUSTOM", custom)
    # Generator Loading

    with dnnlib.util.open_url(pkl, verbose=False) as f:
        data = legacy.load_network_pkl(f, custom=custom)
    g_ema = data["G_ema"].train().requires_grad_(True).to(device)

    # Generator Scoring
    start_time = time.time()
    grad_score_list = Get_Content_Aware_Pruning_Score(generator=g_ema,
                                                      n_sample=n_samples,
                                                      batch_size=batch_size,
                                                      noise_prob=noise_prob,
                                                      device=device, info_print=info_print)

    grad_score_array = np.array([np.array(grad_score) for grad_score in grad_score_list])
    content_aware_pruning_score = np.sum(grad_score_array, axis=0)

    end_time = time.time()

    print('The content-aware metric scoring takes: ' + str(round(end_time - start_time, 4)) + ' seconds')

    # Generator Pruning
    net_shape = Get_Network_Shape(g_ema)
    rmve_list = Get_Uniform_RmveList(net_shape, remove_ratio)
    prune_net_mask = Generate_Prune_Mask_List(content_aware_pruning_score, net_shape, rmve_list,
                                              info_print=info_print)

    shapes = [mask[mask].shape for mask in prune_net_mask]
    channels_dict = {2**(i+2):shape[0] for i, shape in enumerate(shapes[::2])}
    pruned_generator_dict = Mask_the_Generator(g_ema, prune_net_mask)
    from architectures.custom_stylegan2 import Generator
    d_G = Generator(g_ema.z_dim, g_ema.c_dim, g_ema.w_dim, g_ema.img_resolution, g_ema.img_channels,
                    channels_dict=channels_dict)
    d_G.load_state_dict(pruned_generator_dict)
    z = torch.randn(4, 512).to(device)
    og_img, _ = g_ema(z, None)
    pruned_img, _ = d_G.cuda()(z, None)
    torchvision.utils.save_image(og_img,"og_img.png")
    torchvision.utils.save_image(pruned_img,"pruned_img.png")

    snapshot_data = dict(G=d_G, D=data["D"], G_ema=d_G, augment_pipe=data["augment_pipe"],
                         training_set_kwargs=data["training_set_kwargs"])
    for key, value in snapshot_data.items():
        if isinstance(value, torch.nn.Module):
            value = copy.deepcopy(value).eval().requires_grad_(False)
            snapshot_data[key] = value.cpu()
        del value  # conserve memory')

    m_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    path = outdir
    ckpt_file = path + 'content_aware_pruned_' + str(remove_ratio) + '_' + str(
        g_ema.img_resolution) + 'px_model_' + m_time + '.pkl'
    if not os.path.exists(path):
        os.makedirs(path)
    with open(ckpt_file, 'wb') as f:
        pickle.dump(snapshot_data, f)


if __name__ == "__main__":
    clickmain()
