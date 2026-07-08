# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import importlib
import logging
import os
import traceback

import torch
import torch.utils.cpp_extension

from torch_utils import build_cache

#----------------------------------------------------------------------------
# Global options.

verbosity = 'brief' # Verbosity level: 'none', 'brief', 'full'

#----------------------------------------------------------------------------
# Main entry point for compiling and loading C++/CUDA plugins.

_cached_plugins = dict()

def get_plugin(module_name, sources, headers=None, source_dir=None, **build_kwargs):
    assert verbosity in ['none', 'brief', 'full']
    if headers is None:
        headers = []
    if source_dir is not None:
        sources = [os.path.join(source_dir, fname) for fname in sources]
        headers = [os.path.join(source_dir, fname) for fname in headers]

    # Already cached?
    if module_name in _cached_plugins:
        return _cached_plugins[module_name]

    # Print status.
    if verbosity == 'full':
        print(f'Setting up PyTorch plugin "{module_name}"...')
    elif verbosity == 'brief':
        print(f'Setting up PyTorch plugin "{module_name}"... ', end='', flush=True)
    verbose_build = (verbosity == 'full')

    # Compile and load.
    try: # pylint: disable=too-many-nested-blocks
        # Build into the shared cache keyed by (source digest, torch version,
        # compute capabilities). Sources are copied into the cache entry so
        # file timestamps and names stay stable across builds, allowing fast
        # incremental rebuilds (and working around the *.cu dependency bug
        # in ninja config). This requires all source files to reside in a
        # single directory (just for simplicity).
        all_source_files = sorted(sources + headers)
        all_source_dirs = set(os.path.dirname(fname) for fname in all_source_files)
        if len(all_source_dirs) == 1:
            capability = build_cache.current_capability()
            # Any complete entry covering the current device works, including
            # multi-arch entries produced by scripts/precompile_ops.py.
            cached_build_dir = build_cache.find_compatible_build_dir(module_name, all_source_files, capability, verbose=verbose_build)
            if cached_build_dir is None:
                cached_build_dir = build_cache.plugin_build_dir(module_name, all_source_files, [capability], verbose=verbose_build)
                # Failed or abandoned entries poison the cache; remove them
                # before building. The compiler is only required on this path.
                build_cache.clean_failed_build(cached_build_dir, module_name)
                build_cache.setup_compiler_env()
                build_cache.warn_toolkit_mismatch()
                build_cache.pin_arch_list()
                build_cache.populate_sources(cached_build_dir, all_source_files)
                cached_sources = [os.path.join(cached_build_dir, os.path.basename(fname)) for fname in sources]
                torch.utils.cpp_extension.load(name=module_name, build_directory=cached_build_dir,
                    verbose=verbose_build, sources=cached_sources, **build_kwargs)
                build_cache.mark_complete(cached_build_dir, module_name)
            # A complete entry (JIT-warmed or precompiled) loads directly,
            # without compiler or CUDA toolkit access.
            module = build_cache.import_from_cache(module_name, cached_build_dir)
        else:
            build_cache.setup_compiler_env()
            build_cache.warn_toolkit_mismatch()
            build_cache.pin_arch_list()
            torch.utils.cpp_extension.load(name=module_name, verbose=verbose_build, sources=sources, **build_kwargs)
            module = importlib.import_module(module_name)

    except Exception:
        if verbosity == 'brief':
            print('Failed!')
        logging.getLogger(__name__).warning(
            'Failed to set up PyTorch plugin "%s"; the ops that use it fall back to their slow '
            'reference implementation. Details:\n\n%s', module_name, traceback.format_exc())
        _cached_plugins[module_name] = None
        return None

    # Print status and add to cache dict.
    if verbosity == 'full':
        print(f'Done setting up PyTorch plugin "{module_name}".')
    elif verbosity == 'brief':
        print('Done.')
    _cached_plugins[module_name] = module
    return module

#----------------------------------------------------------------------------
