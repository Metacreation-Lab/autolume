import datetime

import torch
import numpy as np
from tqdm import trange

from ganspace import estimators


def setup_estimator(name, num_features, alpha):
    return estimators.get_estimator(name, num_features, alpha)

def sample_latent(n_latents, model, device, c=None, project=False):
    latent = torch.randn(n_latents, model.z_dim,device=device)
    if project:
        latent = model.mapping(latent, c)[:, 0]
    return latent

def get_max_batch_size(model, device):
    # Reset statistics
    torch.cuda.reset_max_memory_cached(device)
    torch.cuda.reset_max_memory_allocated(device)
    total_mem = torch.cuda.get_device_properties(device).total_memory
    B_max = 20
    # Measure actual usage
    for i in range(2, B_max, 2):
        z = torch.randn(i, model.z_dim).to(device)
        model.mapping.forward(z, None)
        maxmem = torch.cuda.max_memory_allocated(device)
        del z
        if maxmem > 0.5*total_mem:
            print('Batch size {:d}: memory usage {:.0f}MB'.format(i, maxmem / 1e6))
            return i
    return B_max

def fit(queue, reply):
    name, num_features, model, device, project, alpha = queue.get()
    while queue.qsize() > 0:
        name, num_features, model, device, project, alpha = queue.get()


    sample_shape = model.w_dim
    sample_dims = np.prod(sample_shape)
    print('Feature shape:', sample_shape, sample_dims)
    input_shape = model.z_dim
    input_dims = np.prod(input_shape)
    print('Input shape:', input_shape, input_dims)
    reply.put(["Setting up estimator", (None, None), False])
    transformer = setup_estimator(name, num_features, alpha)
    reply.put(["Estimating Batch Size", (None, None), False])
    B = get_max_batch_size(model, device)
    # Divisible by B (ignored in output name)
    N = 300000 // B * B

    # Compute maximum batch size based on RAM + pagefile budget
    target_bytes = 20 * 1_000_000_000  # GB
    feat_size_bytes = sample_dims * np.dtype('float64').itemsize
    N_limit_RAM = np.floor_divide(target_bytes, feat_size_bytes)
    if not transformer.batch_support and N > N_limit_RAM:
        print('WARNING: estimator does not support batching, ' \
              'given config will use {:.1f} GB memory.'.format(feat_size_bytes / 1_000_000_000 * N))
    reply.put(['B={}, N={}, dims={}, N/dims={:.1f}'.format(B, N, sample_dims, N / sample_dims), (None, None), False])


    # Must not depend on chosen batch size (reproducibility)
    NB = max(B, max(2_000, 3 * 80))  # ipca: as large as possible!

    samples = None
    if not transformer.batch_support:
        samples = np.zeros((N + NB, sample_dims), dtype=np.float32)
    n_lat = ((N + NB - 1) // B + 1) * B
    n_lat = ((N + NB - 1) // B + 1) * B
    latents = np.zeros((n_lat, input_shape), dtype=np.float32)

    # Sampling latents for pca (DSET)
    reply.put(("Sampling Latents", (None, None), False))
    with torch.no_grad():
        for i in trange(n_lat // B, desc='Sampling latents'):
            reply.put(["Sampling Latents: " + str(i) + " of " + str(n_lat // B) + " batches", (None, None), False])
            latents[i * B:(i + 1) * B] = sample_latent(B, model, device, None, project).cpu().numpy()
    #FITTING
    X = np.ones((NB, sample_dims), dtype=np.float32)
    action = 'Fitting' if transformer.batch_support else 'Collecting'
    for gi in trange(0, N, NB, desc=f'{action} batches (NB={NB})', ascii=True):
        reply.put([f"{action} Batches: {gi} of {N} batches", (None, None), False])
        for mb in range(0, NB, B):
            z = latents[gi + mb:gi + mb + B]
            batch = z.reshape((B, -1))
            space_left = min(B, NB - mb)
            X[mb:mb+space_left] = batch[:space_left]
        if transformer.batch_support:
            if not transformer.fit_partial(X.reshape(-1, sample_dims)):
                break
        else:
            samples[gi:gi + NB, :] = X.copy()

    timestamp = lambda: datetime.datetime.now().strftime("%d.%m %H:%M")

    if not transformer.batch_support:
        X = samples  # Use all samples
        X_global_mean = X.mean(axis=0, keepdims=True, dtype=np.float32)
        X -= X_global_mean

        reply.put([f'[{timestamp()}] Fitting whole batch', (None, None), False])
        t_start_fit = datetime.datetime.now()

        transformer.fit(X)

        reply.put([f'[{timestamp()}] Done in {datetime.datetime.now() - t_start_fit}', (None, None), False])
        assert np.all(transformer.transformer.mean_ < 1e-3), 'Mean of normalized data should be zero'
    else:
        X_global_mean = transformer.transformer.mean_.reshape((1, sample_dims))
        X = X.reshape(-1, sample_dims)
        X -= X_global_mean

    X_comp, X_stdev, X_var_ratio = transformer.get_components()

    assert X_comp.shape[1] == sample_dims \
           and X_comp.shape[0] == num_features \
           and X_global_mean.shape[1] == sample_dims \
           and X_stdev.shape[0] == num_features, 'Invalid shape'

    # 'Activations' are really latents in a secondary latent space
    Z_comp = X_comp
    Z_global_mean = X_global_mean

    # Normalize
    Z_comp /= np.linalg.norm(Z_comp, axis=-1, keepdims=True)

    torch.cuda.empty_cache()
    reply.put(("", (X_comp, Z_comp), True))



