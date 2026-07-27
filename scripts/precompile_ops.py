"""Precompile the custom C++/CUDA ops into the shared build cache.

Warms exactly the cache that torch_utils.ops.custom_ops loads from: after
precompiling on a machine with a full CUDA/MSVC toolchain, any environment
sharing the cache (same torch, CUDA and Python versions) loads the ops
without compiler access. Drives torch.utils.cpp_extension's ninja build
directly instead of custom_ops.get_plugin so that a single process can
build for many compute capabilities without importing the built modules.

Each plugin is built once as a single fatbin entry covering all requested
compute capabilities.

The cache lives at <repo>/torch_extensions, where the app loads it from in
development mode; set TORCH_EXTENSIONS_DIR to relocate it. --export copies
the entries stripped down to the loadable artifacts, ready to ship in the
release bundle.

Examples:

\b
# Precompile for the release archs (RTX 20/30/40/50 series).
python scripts/precompile_ops.py

\b
# Precompile for explicit compute capabilities.
python scripts/precompile_ops.py --arch=8.6,8.9
"""

import os
import shutil
import sys

_REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_DIR)
os.environ.setdefault('TORCH_EXTENSIONS_DIR', os.path.join(_REPO_DIR, 'torch_extensions'))

import click
import torch
import torch.utils.cpp_extension

from torch_utils import build_cache

DEFAULT_ARCHS = '7.5,8.6,8.9,12.0'

#----------------------------------------------------------------------------
# Keep in sync with the _init() calls in
# torch_utils/ops/{bias_act,filtered_lrelu,upfirdn2d}.py.

_OPS_DIR = os.path.join(_REPO_DIR, 'src', 'torch_utils', 'ops')

PLUGIN_SPECS = [
    dict(module_name='bias_act_plugin',
         sources=['bias_act.cpp', 'bias_act.cu'],
         headers=['bias_act.h'],
         extra_cuda_cflags=['--use_fast_math', '--allow-unsupported-compiler']),
    dict(module_name='filtered_lrelu_plugin',
         sources=['filtered_lrelu.cpp', 'filtered_lrelu_wr.cu', 'filtered_lrelu_rd.cu', 'filtered_lrelu_ns.cu'],
         headers=['filtered_lrelu.h', 'filtered_lrelu.cu'],
         extra_cuda_cflags=['--use_fast_math', '--allow-unsupported-compiler']),
    dict(module_name='upfirdn2d_plugin',
         sources=['upfirdn2d.cpp', 'upfirdn2d.cu'],
         headers=['upfirdn2d.h'],
         extra_cuda_cflags=['--use_fast_math', '--allow-unsupported-compiler']),
]

#----------------------------------------------------------------------------

def build_plugin(spec, capabilities, verbose):
    module_name = spec['module_name']
    sources = [os.path.join(_OPS_DIR, fname) for fname in spec['sources']]
    headers = [os.path.join(_OPS_DIR, fname) for fname in spec['headers']]
    all_source_files = sorted(sources + headers)

    build_dir = build_cache.plugin_build_dir(module_name, all_source_files, capabilities, verbose=verbose)
    if build_cache.is_complete(build_dir, module_name):
        return 'cached', build_dir

    build_cache.clean_failed_build(build_dir, module_name)
    build_cache.populate_sources(build_dir, all_source_files)
    build_cache.pin_arch_list(';'.join(capabilities))
    cached_sources = [os.path.join(build_dir, os.path.basename(fname)) for fname in sources]
    torch.utils.cpp_extension._write_ninja_file_and_build_library( # pylint: disable=protected-access
        name=module_name,
        sources=cached_sources,
        extra_cflags=[],
        extra_cuda_cflags=spec['extra_cuda_cflags'],
        extra_sycl_cflags=[],
        extra_ldflags=[],
        extra_include_paths=[],
        build_directory=build_dir,
        verbose=verbose,
        with_cuda=True,
        with_sycl=False)
    build_cache.mark_complete(build_dir, module_name)
    return 'built', build_dir

#----------------------------------------------------------------------------

def export_entry(build_dir, module_name, export_dir):
    """Copy a complete cache entry keeping only what import_from_cache needs:
    the compiled artifact and the completion marker."""
    entry_dir = os.path.join(export_dir, module_name, os.path.basename(build_dir))
    os.makedirs(entry_dir, exist_ok=True)
    artifact = build_cache._artifact_path(build_dir, module_name) # pylint: disable=protected-access
    shutil.copyfile(artifact, os.path.join(entry_dir, os.path.basename(artifact)))
    marker = os.path.join(build_dir, build_cache._COMPLETE_MARKER) # pylint: disable=protected-access
    shutil.copyfile(marker, os.path.join(entry_dir, os.path.basename(marker)))

#----------------------------------------------------------------------------

@click.command(help=__doc__)
@click.option('--arch', 'archs', default=DEFAULT_ARCHS, show_default=True,
              help='Comma-separated compute capabilities to build for.')
@click.option('--export', 'export_dir', type=click.Path(file_okay=False),
              help='Also copy the built entries (artifact + completion marker only) into this directory.')
@click.option('--allow-toolkit-mismatch', is_flag=True,
              help='Build even if the CUDA toolkit major version differs from the torch CUDA runtime.')
@click.option('--verbose', is_flag=True, help='Show build output.')
def main(archs, export_dir, allow_toolkit_mismatch, verbose):
    if torch.version.cuda is None:
        raise click.ClickException('CUDA-enabled torch build required.')
    archs = [arch.strip() for arch in archs.replace(';', ',').split(',') if arch.strip()] or [build_cache.current_capability()]
    build_cache.setup_compiler_env()
    mismatch = build_cache.toolkit_mismatch()
    if mismatch is not None:
        toolkit, runtime = mismatch
        if not allow_toolkit_mismatch:
            raise click.ClickException(
                f'CUDA toolkit {toolkit} does not match the torch CUDA runtime {runtime}. Ops precompiled this '
                f'way embed the CUDA {toolkit.split(".")[0]} runtime and need a correspondingly recent GPU driver '
                f'on every machine that loads this cache. Set CUDA_HOME/CUDA_PATH to a CUDA '
                f'{runtime.split(".")[0]}.x toolkit, or pass --allow-toolkit-mismatch to build anyway.')
        build_cache.warn_toolkit_mismatch()
    for spec in PLUGIN_SPECS:
        print(f'Building {spec["module_name"]} for sm{"_".join(sorted(archs, key=float))}... ', end='', flush=True)
        status, build_dir = build_plugin(spec, archs, verbose)
        print(status)
        if export_dir:
            export_entry(build_dir, spec['module_name'], export_dir)
    if export_dir:
        print(f'Exported entries to {export_dir}')

#----------------------------------------------------------------------------

if __name__ == '__main__':
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
