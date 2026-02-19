import dnnlib
from torch_utils import legacy
import torchvision
import torch
import numpy as np
from ganspace.extract_pca import fit

# Auto-detect best available device
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")
with dnnlib.util.open_url("/home/olaf/PycharmProjects/Autolume_Live_2/models/ffhq.pkl", verbose=False) as f:
    data = legacy.load_network_pkl(f, custom=True)
G = data["G"].to(device)

x_comp, z_comp = fit("ipca", 12, G, device, project=True)
print(x_comp.shape, z_comp.shape)

import gc
import os

z = torch.randn(4, 512)
for i, comp in enumerate(x_comp):
    print(i)
    os.makedirs(f'out/Z/w/{i}/', exist_ok=True)
    for j, alpha in enumerate(np.linspace(-3, 3, 50)):
        gc.collect()
        print(alpha)
        w = G.mapping(z.to(device), None)
        aug = w + (alpha * torch.from_numpy(comp).to(device))  # z.to(device)
        out = G.synthesis(aug)
        img = ((out + 1) / 2).clamp(0, 1)
        torchvision.utils.save_image(img.cpu(), f'out/Z/w/{i}/{j}_test.png')