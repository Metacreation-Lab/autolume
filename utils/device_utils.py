"""Shared helpers for selecting and managing the torch compute device.

All code that needs a torch device should call get_device() instead of
hardcoding 'cuda', so the same code paths run on CUDA (Windows/Linux),
MPS (Apple Silicon), and CPU.
"""

import sys

import torch

_device = None


def get_device():
    """Return the best available torch device: CUDA, then MPS, then CPU. Cached."""
    global _device
    if _device is None:
        if torch.cuda.is_available():
            _device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            _device = torch.device('mps')
        else:
            _device = torch.device('cpu')
    return _device


def synchronize(device=None):
    """Block until all queued work on the device has finished. No-op on CPU."""
    device = torch.device(device) if device is not None else get_device()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()


def empty_cache(device=None):
    """Release cached device memory back to the allocator. No-op on CPU."""
    device = torch.device(device) if device is not None else get_device()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()


def is_macos():
    return sys.platform == 'darwin'
