#!/usr/bin/env python
"""Probe MPS support for the operations StyleGAN training needs.

Run on Apple Silicon from the repo root:

    uv run probe_mps_training_ops.py

Each probe reports one of:
  NATIVE        ran entirely on MPS
  CPU-FALLBACK  ran correctly, but the listed ops fell back to the CPU
  FAIL          raised; the first line of the error is shown

Caveat: PyTorch emits the fallback warning once per operator per process, so
a fallback op is attributed to the first probe that hits it. Probe order goes
from narrow to integrated for that reason.

See TRAINING_MPS.md for how these results feed the plan.
"""

import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import re
import sys
import warnings

import torch

RESULTS = []


def run_probe(name, fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            fn()
        except Exception as e:
            detail = ' '.join(str(e).split())[:160]
            RESULTS.append((name, 'FAIL', detail))
            return
    ops = set()
    for w in caught:
        msg = str(w.message)
        if 'MPS' in msg and 'fall back' in msg:
            m = re.search(r"operator '([^']+)'", msg)
            ops.add(m.group(1) if m else msg[:60])
    if ops:
        RESULTS.append((name, 'CPU-FALLBACK', ', '.join(sorted(ops))))
    else:
        RESULTS.append((name, 'NATIVE', ''))


def main():
    if not torch.backends.mps.is_available():
        print('MPS is not available on this machine; nothing to probe.')
        return 1
    dev = torch.device('mps')
    print(f'torch {torch.__version__} | '
          f'PYTORCH_ENABLE_MPS_FALLBACK={os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")}')
    print()

    # 1. float64 storage (expected FAIL: documents why training_stats detours via CPU).
    def float64_cast():
        torch.ones(4, device=dev).to(torch.float64)
    run_probe('float64 cast on mps (expected FAIL)', float64_cast)

    # 2. training_stats moments with the CPU detour (the shipped fix).
    def stats_detour():
        from torch_utils import training_stats
        collector = training_stats.Collector(regex='Probe/.*')
        training_stats.report('Probe/loss', torch.randn(64, device=dev))
        collector.update()
        assert collector.as_dict()['Probe/loss'].num == 64
    run_probe('training_stats report/collect on mps', stats_detour)

    # 3. grid_sample (ADA geometric path) first-order backward.
    from torch_utils.ops import grid_sample_gradfix
    grid_sample_gradfix.enabled = True

    def make_grid(batch):
        theta = torch.eye(2, 3, device=dev).unsqueeze(0).repeat(batch, 1, 1)
        return torch.nn.functional.affine_grid(theta, [batch, 3, 16, 16], align_corners=False)

    def gs_first():
        x = torch.randn(2, 3, 16, 16, device=dev, requires_grad=True)
        y = grid_sample_gradfix.grid_sample(x, make_grid(2))
        y.sum().backward()
    run_probe('grid_sample first-order backward', gs_first)

    # 4. grid_sample second-order (R1 flowing back through the augment pipe).
    def gs_second():
        x = torch.randn(2, 3, 16, 16, device=dev, requires_grad=True)
        y = grid_sample_gradfix.grid_sample(x, make_grid(2))
        g = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
        g.square().sum().backward()
    run_probe('grid_sample r1-style second-order', gs_second)

    # 5. Convolution double backward (R1 on a conv discriminator).
    def conv_r1():
        net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1), torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(16, 16, 3, padding=1, stride=2), torch.nn.LeakyReLU(0.2),
            torch.nn.Flatten(), torch.nn.Linear(16 * 8 * 8, 1)).to(dev)
        x = torch.randn(4, 3, 16, 16, device=dev, requires_grad=True)
        logits = net(x).sum()
        g = torch.autograd.grad(logits, x, create_graph=True)[0]
        (g.square().sum() * 5).backward()
    run_probe('conv double backward (r1 micro-step)', conv_r1)

    # 6. Path-length style second order wrt an intermediate activation.
    def pl_reg():
        mapping = torch.nn.Linear(64, 64).to(dev)
        synth = torch.nn.Sequential(
            torch.nn.Linear(64, 16 * 8 * 8), torch.nn.Unflatten(1, (16, 8, 8)),
            torch.nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1)).to(dev)
        z = torch.randn(4, 64, device=dev)
        ws = mapping(z)
        img = synth(ws)
        pl_noise = torch.randn_like(img) / (img.shape[2] * img.shape[3]) ** 0.5
        pl_grads = torch.autograd.grad((img * pl_noise).sum(), ws, create_graph=True)[0]
        pl_grads.square().sum(1).sqrt().mean().backward()
    run_probe('path-length second order (pl micro-step)', pl_reg)

    # 7. ADA sampling math and the upfirdn2d reference path.
    def ada_misc():
        from torch_utils.ops import upfirdn2d
        f = upfirdn2d.setup_filter([1, 3, 3, 1], device=dev)
        x = torch.randn(2, 3, 16, 16, device=dev)
        upfirdn2d.upsample2d(x, f)
        upfirdn2d.downsample2d(x, f)
        torch.exp2(torch.randn(64, device=dev))
        torch.erfinv(torch.rand(64, device=dev) * 0.98 + 0.01)
    run_probe('ada sampling + upfirdn2d reference ops', ada_misc)

    # 8. DiffAugment policies.
    def diffaug():
        from training.DiffAugment_pytorch import DiffAugment
        x = torch.randn(4, 3, 32, 32, device=dev, requires_grad=True)
        y = DiffAugment(x, policy='color,translation,cutout')
        y.sum().backward()
    run_probe('diffaugment color,translation,cutout', diffaug)

    # 9. EMA and grad hygiene ops from the loop.
    def ema_misc():
        a = torch.randn(1000, device=dev)
        b = torch.randn(1000, device=dev)
        b.copy_(a.lerp(b, 0.5))
        from torch_utils import misc
        flat = torch.randn(1000, device=dev)
        misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
    run_probe('ema lerp/copy + nan_to_num', ema_misc)

    # 10. Integrated one-batch G/D step with the real classes, mirroring
    #     train.py's stylegan2 config at 64x64 with the bgc augment pipe.
    def integrated():
        import dnnlib
        from training.losses.scratch_loss import StyleGAN2Loss
        G_kwargs = dnnlib.EasyDict(class_name='architectures.custom_stylegan2.Generator',
                                   z_dim=64, w_dim=64, mapping_kwargs=dnnlib.EasyDict(num_layers=2),
                                   channel_base=2048, channel_max=64,
                                   fused_modconv_default='inference_only')
        D_kwargs = dnnlib.EasyDict(class_name='architectures.custom_stylegan2.Discriminator',
                                   block_kwargs=dnnlib.EasyDict(freeze_layers=0),
                                   mapping_kwargs=dnnlib.EasyDict(),
                                   epilogue_kwargs=dnnlib.EasyDict(mbstd_group_size=2),
                                   channel_base=2048, channel_max=64)
        common = dict(c_dim=0, img_resolution=64, img_channels=3)
        G = dnnlib.util.construct_class_by_name(**G_kwargs, **common).train().requires_grad_(False).to(dev)
        D = dnnlib.util.construct_class_by_name(**D_kwargs, **common).train().requires_grad_(False).to(dev)
        augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe',
                                         xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1,
                                         brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1)
        augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(dev)
        augment_pipe.p.copy_(torch.as_tensor(0.5))
        loss = StyleGAN2Loss(device=dev, G=G, D=D, T=None, augment_pipe=augment_pipe, r1_gamma=10,
                             style_mixing_prob=0.9, pl_weight=2, pl_no_weight_grad=True)
        batch = 4
        z = torch.randn(batch, 64, device=dev)
        c = torch.zeros(batch, 0, device=dev)
        real = torch.randn(batch, 3, 64, 64, device=dev)
        for phase, module in [('Gmain', G), ('Greg', G), ('Dmain', D), ('Dreg', D)]:
            module.requires_grad_(True)
            loss.accumulate_gradients(phase=phase, real_img=real, real_c=c, gen_z=z, gen_c=c,
                                      gain=1, cur_nimg=0)
            module.requires_grad_(False)
        torch.mps.synchronize()
    run_probe('integrated one-batch G/D step (real classes, ada bgc)', integrated)

    print(f'{"probe":<55} {"result":<13} detail')
    print('-' * 100)
    for name, status, detail in RESULTS:
        print(f'{name:<55} {status:<13} {detail}')
    print()

    fallback_ops = sorted({op for _, s, d in RESULTS if s == 'CPU-FALLBACK' for op in d.split(', ')})
    if fallback_ops:
        print('Ops running on CPU via fallback:', ', '.join(fallback_ops))
    unexpected_fails = [n for n, s, _ in RESULTS if s == 'FAIL' and 'expected FAIL' not in n]
    if unexpected_fails:
        print('UNEXPECTED FAILURES:', '; '.join(unexpected_fails))
        return 1
    print('No unexpected failures. See TRAINING_MPS.md phase 2 for how to act on fallbacks.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
