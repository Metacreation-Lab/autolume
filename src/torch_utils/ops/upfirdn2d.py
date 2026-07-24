# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Custom PyTorch ops for efficient resampling of 2D images."""

import os
import weakref
import numpy as np
import torch

from .. import custom_ops
from .. import misc
from . import conv2d_gradfix

#----------------------------------------------------------------------------

_plugin = None

def _init():
    global _plugin
    if _plugin is None:
        _plugin = custom_ops.get_plugin(
            module_name='upfirdn2d_plugin',
            sources=['upfirdn2d.cpp', 'upfirdn2d.cu'],
            headers=['upfirdn2d.h'],
            source_dir=os.path.dirname(__file__),
            extra_cuda_cflags=['--use_fast_math', '--allow-unsupported-compiler'],
        )
    return _plugin is not None

def _parse_scaling(scaling):
    if isinstance(scaling, int):
        scaling = [scaling, scaling]
    assert isinstance(scaling, (list, tuple))
    assert all(isinstance(x, int) for x in scaling)
    sx, sy = scaling
    assert sx >= 1 and sy >= 1
    return sx, sy

def _parse_padding(padding):
    if isinstance(padding, int):
        padding = [padding, padding]
    assert isinstance(padding, (list, tuple))
    assert all(isinstance(x, int) for x in padding)
    if len(padding) == 2:
        padx, pady = padding
        padding = [padx, padx, pady, pady]
    padx0, padx1, pady0, pady1 = padding
    return padx0, padx1, pady0, pady1

def _get_filter_size(f):
    if f is None:
        return 1, 1
    assert isinstance(f, torch.Tensor) and f.ndim in [1, 2]
    fw = f.shape[-1]
    fh = f.shape[0]
    with misc.suppress_tracer_warnings():
        fw = int(fw)
        fh = int(fh)
    misc.assert_shape(f, [fh, fw][:f.ndim])
    assert fw >= 1 and fh >= 1
    return fw, fh

#----------------------------------------------------------------------------

def setup_filter(f, device=torch.device('cpu'), normalize=True, flip_filter=False, gain=1, separable=None):
    r"""Convenience function to setup 2D FIR filter for `upfirdn2d()`.

    Args:
        f:           Torch tensor, numpy array, or python list of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable),
                     `[]` (impulse), or
                     `None` (identity).
        device:      Result device (default: cpu).
        normalize:   Normalize the filter so that it retains the magnitude
                     for constant input signal (DC)? (default: True).
        flip_filter: Flip the filter? (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).
        separable:   Return a separable filter? (default: select automatically).

    Returns:
        Float32 tensor of the shape
        `[filter_height, filter_width]` (non-separable) or
        `[filter_taps]` (separable).
    """
    # Validate.
    if f is None:
        f = 1
    f = torch.as_tensor(f, dtype=torch.float32)
    assert f.ndim in [0, 1, 2]
    assert f.numel() > 0
    if f.ndim == 0:
        f = f[np.newaxis]

    # Separable?
    if separable is None:
        separable = (f.ndim == 1 and f.numel() >= 8)
    if f.ndim == 1 and not separable:
        f = f.ger(f)
    assert f.ndim == (1 if separable else 2)

    # Apply normalize, flip, gain, and device.
    if normalize:
        f /= f.sum()
    if flip_filter:
        f = f.flip(list(range(f.ndim)))
    f = f * (gain ** (f.ndim / 2))
    f = f.to(device=device)
    return f

#----------------------------------------------------------------------------

def upfirdn2d(x, f, up=1, down=1, padding=0, flip_filter=False, gain=1, impl='cuda'):
    r"""Pad, upsample, filter, and downsample a batch of 2D images.

    Performs the following sequence of operations for each channel:

    1. Upsample the image by inserting N-1 zeros after each pixel (`up`).

    2. Pad the image with the specified number of zeros on each side (`padding`).
       Negative padding corresponds to cropping the image.

    3. Convolve the image with the specified 2D FIR filter (`f`), shrinking it
       so that the footprint of all output pixels lies within the input image.

    4. Downsample the image by keeping every Nth pixel (`down`).

    This sequence of operations bears close resemblance to scipy.signal.upfirdn().
    The fused op is considerably more efficient than performing the same calculation
    using standard PyTorch ops. It supports gradients of arbitrary order.

    Args:
        x:           Float32/float64/float16 input tensor of the shape
                     `[batch_size, num_channels, in_height, in_width]`.
        f:           Float32 FIR filter of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        up:          Integer upsampling factor. Can be a single int or a list/tuple
                     `[x, y]` (default: 1).
        down:        Integer downsampling factor. Can be a single int or a list/tuple
                     `[x, y]` (default: 1).
        padding:     Padding with respect to the upsampled image. Can be a single number
                     or a list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`
                     (default: 0).
        flip_filter: False = convolution, True = correlation (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).
        impl:        Implementation to use. Can be `'ref'` or `'cuda'` (default: `'cuda'`).

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
    """
    assert isinstance(x, torch.Tensor)
    assert impl in ['ref', 'cuda']
    if impl == 'cuda' and x.device.type == 'cuda' and _init():
        return _upfirdn2d_cuda(up=up, down=down, padding=padding, flip_filter=flip_filter, gain=gain).apply(x, f)
    return _upfirdn2d_ref(x, f, up=up, down=down, padding=padding, flip_filter=flip_filter, gain=gain)

#----------------------------------------------------------------------------

# Cache of per-filter execution plans, keyed by id(filter) with weakref
# validation so that id reuse after garbage collection cannot return stale
# plans. Building a plan reads the filter values (a device sync for a filter
# resident on an accelerator), so it must happen only once per unique filter.
_filter_plan_cache = dict()  # id(f) -> (weakref(f), {gain: plan})

def _build_filter_plan(f, gain):
    """Decompose the filter for the fast path.

    Returns ('sep', fa, fb, fa_flip, fb_flip) for rank-1 filters, where fa/fb
    are the H-axis and W-axis 1D taps with sqrt(gain) folded into each, or
    ('dense', k, None, k_flip, None) for genuinely non-separable 2D filters
    with gain folded in. All tensors are float32 on f.device.
    """
    fc = f.detach().cpu().to(torch.float64)
    if fc.ndim == 1:
        a = (fc * (float(gain) ** 0.5)).to(dtype=torch.float32, device=f.device)
        return 'sep', a, a, a.flip(0), a.flip(0)
    s = torch.linalg.svdvals(fc)
    if s.numel() > 1 and bool(s[1] > s[0] * 1e-6):
        k = (fc * float(gain)).to(dtype=torch.float32, device=f.device)
        return 'dense', k, None, k.flip([0, 1]), None
    u, sv, vh = torch.linalg.svd(fc)
    g = sv[0].sqrt() * (float(gain) ** 0.5)
    a = (u[:, 0] * g).to(dtype=torch.float32, device=f.device)
    b = (vh[0, :] * g).to(dtype=torch.float32, device=f.device)
    return 'sep', a, b, a.flip(0), b.flip(0)

def _filter_plan(f, gain):
    key = float(gain)
    entry = _filter_plan_cache.get(id(f))
    if entry is not None:
        ref, plans = entry
        if ref() is f:
            plan = plans.get(key)
            if plan is None:
                plan = _build_filter_plan(f, gain)
                plans[key] = plan
            return plan
        del _filter_plan_cache[id(f)]
    if len(_filter_plan_cache) > 64:  # prune dead entries
        for k in [k for k, v in _filter_plan_cache.items() if v[0]() is None]:
            del _filter_plan_cache[k]
    plan = _build_filter_plan(f, gain)
    _filter_plan_cache[id(f)] = (weakref.ref(f), {key: plan})
    return plan

def _crop_axis(x, dim, start, end):
    """Crop x along dim to [start, end), zero-padding if the range is out of bounds."""
    n = x.shape[dim]
    left = max(-start, 0)
    right = max(end - n, 0)
    if left or right:
        pad = [0, 0, 0, 0]
        pad[(3 - dim) * 2] = left
        pad[(3 - dim) * 2 + 1] = right
        x = torch.nn.functional.pad(x, pad)
        start += left
        end += left
    slc = [slice(None)] * x.ndim
    slc[dim] = slice(start, end)
    return x[tuple(slc)]

@misc.profiled_function
def _upfirdn2d_ref(x, f, up=1, down=1, padding=0, flip_filter=False, gain=1):
    """Native reference implementation of `upfirdn2d()`.

    This is the path taken on every non-CUDA device (MPS, CPU); CUDA still
    dispatches to the fused custom op above. Rank-1 filters (e.g. the default
    [1,3,3,1] outer products) are detected via SVD and applied as two
    consecutive 1D depthwise convolutions. The upsampling zero-stuffing is
    fused into the filtering step via conv_transpose2d(stride=up), so the
    up-scaled intermediate is never materialized -- this is the change that
    matters on MPS, where that oversized intermediate otherwise churns the
    caching allocator. Padding, gain, and flip semantics match the original
    zero-stuffing reference exactly (verified bit-exact).
    """
    # Validate arguments.
    assert isinstance(x, torch.Tensor) and x.ndim == 4
    if f is None:
        f = torch.ones([1, 1], dtype=torch.float32, device=x.device)
    assert isinstance(f, torch.Tensor) and f.ndim in [1, 2]
    assert f.dtype == torch.float32 and not f.requires_grad
    batch_size, num_channels, in_height, in_width = x.shape
    upx, upy = _parse_scaling(up)
    downx, downy = _parse_scaling(down)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    fw, fh = _get_filter_size(f)

    # Check that upsampled buffer is not smaller than the filter.
    upW = in_width * upx + padx0 + padx1
    upH = in_height * upy + pady0 + pady1
    assert upW >= fw and upH >= fh

    kind, fa, fb, fa_flip, fb_flip = _filter_plan(f, gain)

    # Weight builders. conv_transpose2d computes the adjoint of the reference
    # cross-correlation, so it takes the unflipped filter when flip_filter=False;
    # conv2d takes the flipped one, matching the reference's explicit flip.
    def weight1d(t, t_flip, axis, transpose):
        t = t_flip if bool(flip_filter) == transpose else t
        shape = [1, 1, 1, 1]
        shape[axis] = t.numel()
        return t.to(dtype=x.dtype, device=x.device).reshape(shape).repeat(num_channels, 1, 1, 1)

    def weight2d(k, k_flip, transpose):
        k = k_flip if bool(flip_filter) == transpose else k
        return k.to(dtype=x.dtype, device=x.device).reshape(1, 1, *k.shape).repeat(num_channels, 1, 1, 1)

    if upx == 1 and upy == 1:
        # No upsampling: pad (or crop), then convolve. Symmetric non-negative
        # padding is folded into the convolutions to skip a separate pad pass.
        symmetric = padx0 == padx1 and pady0 == pady1 and padx0 >= 0 and pady0 >= 0
        if not symmetric:
            x = torch.nn.functional.pad(x, [padx0, padx1, pady0, pady1])
        padw = (0, padx0) if symmetric else 0
        padh = (pady0, 0) if symmetric else 0
        if kind == 'sep':
            x = conv2d_gradfix.conv2d(input=x, weight=weight1d(fb, fb_flip, 3, False), padding=padw, groups=num_channels)
            x = conv2d_gradfix.conv2d(input=x, weight=weight1d(fa, fa_flip, 2, False), padding=padh, groups=num_channels)
        else:
            padd = (pady0, padx0) if symmetric else 0
            x = conv2d_gradfix.conv2d(input=x, weight=weight2d(fa, fa_flip, False), padding=padd, groups=num_channels)
    else:
        # Fuse upsampling into the filter via transposed convolution. The
        # transposed-conv output is the zero-stuffed convolution shifted by
        # K-1; crop each axis to [K-1-p0 : in*up + p1) to recover the exact
        # reference indexing.
        if kind == 'sep':
            x = conv2d_gradfix.conv_transpose2d(input=x, weight=weight1d(fb, fb_flip, 3, True), stride=(1, upx), groups=num_channels)
            x = _crop_axis(x, 3, fb.numel() - 1 - padx0, in_width * upx + padx1)
            x = conv2d_gradfix.conv_transpose2d(input=x, weight=weight1d(fa, fa_flip, 2, True), stride=(upy, 1), groups=num_channels)
            x = _crop_axis(x, 2, fa.numel() - 1 - pady0, in_height * upy + pady1)
        else:
            x = conv2d_gradfix.conv_transpose2d(input=x, weight=weight2d(fa, fa_flip, True), stride=(upy, upx), groups=num_channels)
            x = _crop_axis(x, 3, fw - 1 - padx0, in_width * upx + padx1)
            x = _crop_axis(x, 2, fh - 1 - pady0, in_height * upy + pady1)

    # Downsample by throwing away pixels.
    if downx > 1 or downy > 1:
        x = x[:, :, ::downy, ::downx]
    return x

#----------------------------------------------------------------------------

_upfirdn2d_cuda_cache = dict()

def _upfirdn2d_cuda(up=1, down=1, padding=0, flip_filter=False, gain=1):
    """Fast CUDA implementation of `upfirdn2d()` using custom ops.
    """
    # Parse arguments.
    upx, upy = _parse_scaling(up)
    downx, downy = _parse_scaling(down)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)

    # Lookup from cache.
    key = (upx, upy, downx, downy, padx0, padx1, pady0, pady1, flip_filter, gain)
    if key in _upfirdn2d_cuda_cache:
        return _upfirdn2d_cuda_cache[key]

    # Forward op.
    class Upfirdn2dCuda(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, f): # pylint: disable=arguments-differ
            assert isinstance(x, torch.Tensor) and x.ndim == 4
            if f is None:
                f = torch.ones([1, 1], dtype=torch.float32, device=x.device)
            if f.ndim == 1 and f.shape[0] == 1:
                f = f.square().unsqueeze(0) # Convert separable-1 into full-1x1.
            assert isinstance(f, torch.Tensor) and f.ndim in [1, 2]
            y = x
            if f.ndim == 2:
                y = _plugin.upfirdn2d(y, f, upx, upy, downx, downy, padx0, padx1, pady0, pady1, flip_filter, gain)
            else:
                y = _plugin.upfirdn2d(y, f.unsqueeze(0), upx, 1, downx, 1, padx0, padx1, 0, 0, flip_filter, 1.0)
                y = _plugin.upfirdn2d(y, f.unsqueeze(1), 1, upy, 1, downy, 0, 0, pady0, pady1, flip_filter, gain)
            ctx.save_for_backward(f)
            ctx.x_shape = x.shape
            return y

        @staticmethod
        def backward(ctx, dy): # pylint: disable=arguments-differ
            f, = ctx.saved_tensors
            _, _, ih, iw = ctx.x_shape
            _, _, oh, ow = dy.shape
            fw, fh = _get_filter_size(f)
            p = [
                fw - padx0 - 1,
                iw * upx - ow * downx + padx0 - upx + 1,
                fh - pady0 - 1,
                ih * upy - oh * downy + pady0 - upy + 1,
            ]
            dx = None
            df = None

            if ctx.needs_input_grad[0]:
                dx = _upfirdn2d_cuda(up=down, down=up, padding=p, flip_filter=(not flip_filter), gain=gain).apply(dy, f)

            assert not ctx.needs_input_grad[1]
            return dx, df

    # Add to cache.
    _upfirdn2d_cuda_cache[key] = Upfirdn2dCuda
    return Upfirdn2dCuda

#----------------------------------------------------------------------------

def filter2d(x, f, padding=0, flip_filter=False, gain=1, impl='cuda'):
    r"""Filter a batch of 2D images using the given 2D FIR filter.

    By default, the result is padded so that its shape matches the input.
    User-specified padding is applied on top of that, with negative values
    indicating cropping. Pixels outside the image are assumed to be zero.

    Args:
        x:           Float32/float64/float16 input tensor of the shape
                     `[batch_size, num_channels, in_height, in_width]`.
        f:           Float32 FIR filter of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        padding:     Padding with respect to the output. Can be a single number or a
                     list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`
                     (default: 0).
        flip_filter: False = convolution, True = correlation (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).
        impl:        Implementation to use. Can be `'ref'` or `'cuda'` (default: `'cuda'`).

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
    """
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    fw, fh = _get_filter_size(f)
    p = [
        padx0 + fw // 2,
        padx1 + (fw - 1) // 2,
        pady0 + fh // 2,
        pady1 + (fh - 1) // 2,
    ]
    return upfirdn2d(x, f, padding=p, flip_filter=flip_filter, gain=gain, impl=impl)

#----------------------------------------------------------------------------

def upsample2d(x, f, up=2, padding=0, flip_filter=False, gain=1, impl='cuda'):
    r"""Upsample a batch of 2D images using the given 2D FIR filter.

    By default, the result is padded so that its shape is a multiple of the input.
    User-specified padding is applied on top of that, with negative values
    indicating cropping. Pixels outside the image are assumed to be zero.

    Args:
        x:           Float32/float64/float16 input tensor of the shape
                     `[batch_size, num_channels, in_height, in_width]`.
        f:           Float32 FIR filter of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        up:          Integer upsampling factor. Can be a single int or a list/tuple
                     `[x, y]` (default: 1).
        padding:     Padding with respect to the output. Can be a single number or a
                     list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`
                     (default: 0).
        flip_filter: False = convolution, True = correlation (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).
        impl:        Implementation to use. Can be `'ref'` or `'cuda'` (default: `'cuda'`).

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
    """
    upx, upy = _parse_scaling(up)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    fw, fh = _get_filter_size(f)
    p = [
        padx0 + (fw + upx - 1) // 2,
        padx1 + (fw - upx) // 2,
        pady0 + (fh + upy - 1) // 2,
        pady1 + (fh - upy) // 2,
    ]
    return upfirdn2d(x, f, up=up, padding=p, flip_filter=flip_filter, gain=gain*upx*upy, impl=impl)

#----------------------------------------------------------------------------

def downsample2d(x, f, down=2, padding=0, flip_filter=False, gain=1, impl='cuda'):
    r"""Downsample a batch of 2D images using the given 2D FIR filter.

    By default, the result is padded so that its shape is a fraction of the input.
    User-specified padding is applied on top of that, with negative values
    indicating cropping. Pixels outside the image are assumed to be zero.

    Args:
        x:           Float32/float64/float16 input tensor of the shape
                     `[batch_size, num_channels, in_height, in_width]`.
        f:           Float32 FIR filter of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        down:        Integer downsampling factor. Can be a single int or a list/tuple
                     `[x, y]` (default: 1).
        padding:     Padding with respect to the input. Can be a single number or a
                     list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`
                     (default: 0).
        flip_filter: False = convolution, True = correlation (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).
        impl:        Implementation to use. Can be `'ref'` or `'cuda'` (default: `'cuda'`).

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
    """
    downx, downy = _parse_scaling(down)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    fw, fh = _get_filter_size(f)
    p = [
        padx0 + (fw - downx + 1) // 2,
        padx1 + (fw - downx) // 2,
        pady0 + (fh - downy + 1) // 2,
        pady1 + (fh - downy) // 2,
    ]
    return upfirdn2d(x, f, down=down, padding=p, flip_filter=flip_filter, gain=gain, impl=impl)

#----------------------------------------------------------------------------
