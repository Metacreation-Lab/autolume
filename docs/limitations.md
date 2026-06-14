# Limitations

This page documents known limitations and platform-specific caveats in Autolume. It is living documentation: the figures and platform notes here are updated as the project evolves and as more hardware is benchmarked.

For the supported, best-performance setup, use Windows or Linux with an NVIDIA GPU (RTX 2070 or higher) and CUDA 12.8+. See the [Windows requirements guide](windows-requirements.md). Everything below describes where behaviour differs from that reference configuration.

## Apple Silicon (macOS / MPS)

**Apple Silicon support is experimental.** Autolume runs natively on M-series Macs through PyTorch's MPS (Metal) backend. The live renderer is fully supported. Training is functional and produces correct results, but it is **significantly slower than on an NVIDIA GPU** and runs in fp32 only. Treat Apple Silicon as a strong platform for rendering and light fine-tuning, not for large or from-scratch training.

### What works

The live visualizer and everything around it run on MPS: model rendering, network bending, model mixing, looping, audio reactivity, OSC, presets, image projection, GANSpace feature extraction, super resolution, screenshots, and recording.

Training (including dataset preparation) is also functional on MPS and is suitable for fine-tuning pretrained models. It is correct, but slow; see below.

### Training performance versus CUDA

MPS trains in fp32. StyleGAN's mixed-precision (fp16) path is tuned for NVIDIA tensor cores and is not enabled on Apple Silicon, and the optimized custom CUDA kernels fall back to slower reference PyTorch operations. As a result, training throughput sits roughly an order of magnitude below a modern NVIDIA card.

An early datapoint (StyleGAN2, 64x64, 500-image dataset, ADA `bgc` pipeline, batch 32, fp32):

| Machine | Throughput | Relative |
|---------|-----------|----------|
| Laptop RTX 4090 (16 GB) | ~14 sec/kimg | 1x |
| Apple M1 Max (64 GB) | ~240 sec/kimg | ~17x slower |

Most of that gap is broad-based: the generator and discriminator convolution passes run about 13-20x slower across the board, which is the expected cost of the hardware and the fp32 / reference-kernel path rather than any single fixable hotspot. One step is disproportionately slow: the R1 gradient penalty runs roughly 67x slower, because ADA's geometric augmentation relies on an operation (`grid_sample`) whose gradient has no Metal implementation and falls back to the CPU, which is especially costly inside R1's second-order (double-backward) computation. R1 runs only periodically, so it is a small share of total time, but it is the one part that is pathologically slow rather than merely slow.

Two caveats on these numbers. They come from a single pair of machines, so treat the ratio as indicative, not exact. And 64x64 is a small, dispatch-bound resolution that penalises MPS more than the 256-512 resolutions artists usually train at, so higher-resolution behaviour, which has not yet been benchmarked, may differ. fp16 is not the missing lever at this resolution: it gave no measurable benefit even on the 4090.

### Practical guidance

- Fine-tune a pretrained model rather than training from scratch.
- Train at 256 or 512 rather than 1024.
- Expect a few-hundred-kimg fine-tune to take from a few hours up to about a day; a full from-scratch run (tens of thousands of kimg) is impractical on Apple Silicon.
- Size the batch to fit comfortably in unified memory, and keep 2 to 4 data loader workers.
- Reduce snapshot frequency: each snapshot deep-copies the generator, discriminator, and EMA generator.

### Memory

Apple Silicon uses unified memory shared between the CPU and GPU; there is no separate VRAM budget. Large resolutions or batch sizes compete with the rest of the system for the same memory, so prefer moderate resolutions and batch sizes.

### Features unavailable on macOS

| Feature | Why |
|---------|-----|
| Model compression / distillation | Relies on the CUDA-only training and pruning stack |
| Full-screen display window | Requires sharing textures across OpenGL profiles, which macOS does not allow |
| Custom CUDA kernel toggle | The custom kernels only compile for CUDA |

These are greyed out in the interface on macOS.

### Planned improvements

This section tracks optimisation work that may narrow the gap; it will be updated as higher-resolution benchmarks are collected.

- Keep R1 on the GPU with a pure-tensor-ops grid sampler, removing the CPU fallback in the augmentation backward.
- Re-evaluate fp16 on MPS at 256 to 512, where it has a better chance of paying off than at 64x64.
- Investigate `torch.compile` with the experimental Metal backend.

The detailed engineering analysis and the original device op-probe results are kept in the project repository in `PORTING_MACOS.md` and `TRAINING_MPS.md`.

_Last reviewed: June 2026._
