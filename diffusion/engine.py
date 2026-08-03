"""Headless StreamDiffusion img2img stage. UI-free so it is unit-testable
and importable on platforms where the streamdiffusion extra is absent."""
import importlib.util

import torch
import torch.nn.functional as F


def is_available():
    return importlib.util.find_spec("streamdiffusion") is not None


def default_params():
    return dict(model="stabilityai/sd-turbo", prompt="", strength=0.5, seed=0,
                lora_path="", lora_scale=1.0, acceleration="none", resolution=512)


def t_indices_for_strength(strength, num_steps=50, steps=1):
    strength = min(max(float(strength), 0.0), 1.0)
    start = round((1.0 - strength) * (num_steps - 1))
    if steps == 1:
        return [start]
    stride = max(1, (num_steps - 1 - start) // steps)
    idx = [min(start + i * stride, num_steps - 1) for i in range(steps)]
    return sorted(set(idx))


def to_diffusion_input(out, resolution):
    x = out[:, :3] if out.shape[1] >= 3 else out[:, :1].repeat(1, 3, 1, 1)
    x = x.to(torch.float32).clamp(-1, 1).add(1).div(2)
    if x.shape[-2:] != (resolution, resolution):
        x = F.interpolate(x, size=(resolution, resolution), mode="bilinear",
                          antialias=True, align_corners=False)
    return x


def from_diffusion_output(img):
    return img.to(torch.float32).clamp(0, 1) * 2 - 1


def build_key(params):
    return (params["model"], params["acceleration"], params["resolution"],
            params["lora_path"], params["lora_scale"])
