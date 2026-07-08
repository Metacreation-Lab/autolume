"""Shared build-cache utilities for the custom C++/CUDA ops.

Both the JIT loader (`torch_utils.custom_ops`) and the standalone
precompiler (`scripts/precompile_ops.py`) build into the same on-disk cache.
A cache entry is keyed by (source digest, Python version, torch version,
compute capabilities) -- never by GPU name -- so entries are shared across
GPUs of the same compute capability and can be produced on one machine and
consumed on another. A single entry may cover several compute capabilities (a fatbin
built by the precompiler); the loader picks any complete entry whose arch
set contains the current device. An entry is only trusted once its
completion marker exists; incomplete entries are treated as failed builds
and deleted before rebuilding. A complete entry is imported directly from
disk, without compiler or CUDA toolkit access.
"""

import glob
import hashlib
import importlib.machinery
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import uuid

import torch
import torch.utils.cpp_extension

_COMPLETE_MARKER = '.build_complete'

#----------------------------------------------------------------------------
# Cache key and directory resolution.

def source_digest(source_files):
    hash_md5 = hashlib.md5()
    for src in sorted(source_files):
        with open(src, 'rb') as f:
            hash_md5.update(f.read())
    return hash_md5.hexdigest()

def current_capability():
    major, minor = torch.cuda.get_device_capability()
    return f'{major}.{minor}'

def _arch_tag(capabilities):
    return 'sm' + '_'.join(sorted(capabilities, key=float))

def _key_prefix(source_files):
    # The Python tag duplicates torch's default py/cu-versioned cache layout,
    # but keeps entries self-contained when TORCH_EXTENSIONS_DIR relocates
    # the cache to a flat directory.
    return f'{source_digest(source_files)}-py{sys.version_info.major}{sys.version_info.minor}-torch{torch.__version__}-'

def plugin_build_dir(module_name, source_files, capabilities, verbose=False):
    assert torch.version.cuda is not None, 'CUDA-enabled torch build required'
    root = torch.utils.cpp_extension._get_build_directory(module_name, verbose=verbose) # pylint: disable=protected-access
    return os.path.join(root, _key_prefix(source_files) + _arch_tag(capabilities))

def find_compatible_build_dir(module_name, source_files, capability, verbose=False):
    """Complete cache entry usable on a device of the given capability:
    the exact single-arch entry if complete, else any complete entry whose
    arch set contains the capability, else None."""
    exact = plugin_build_dir(module_name, source_files, [capability], verbose=verbose)
    if is_complete(exact, module_name):
        return exact
    pattern = os.path.join(os.path.dirname(exact), _key_prefix(source_files) + 'sm*')
    for build_dir in sorted(glob.glob(pattern)):
        archs = os.path.basename(build_dir).rsplit('-sm', 1)[1].split('_')
        if capability in archs and is_complete(build_dir, module_name):
            return build_dir
    return None

#----------------------------------------------------------------------------
# Entry completion state and failed-build cleanup.

def _artifact_path(build_dir, module_name):
    ext = '.pyd' if os.name == 'nt' else '.so'
    return os.path.join(build_dir, module_name + ext)

def is_complete(build_dir, module_name):
    return os.path.isfile(os.path.join(build_dir, _COMPLETE_MARKER)) and os.path.isfile(_artifact_path(build_dir, module_name))

def mark_complete(build_dir, module_name):
    artifact = _artifact_path(build_dir, module_name)
    assert os.path.isfile(artifact), f'build did not produce {artifact}'
    with open(os.path.join(build_dir, _COMPLETE_MARKER), 'w') as f:
        # Provenance, for diagnosing entries built on other machines.
        f.write(f'artifact={os.path.basename(artifact)}\n')
        f.write(f'torch={torch.__version__}\n')
        f.write(f'cuda_runtime={torch.version.cuda}\n')
        f.write(f'cuda_toolkit={toolkit_version() or "unknown"}\n')

def clean_failed_build(build_dir, module_name):
    """Delete an incomplete entry (interrupted or failed build) so the next
    build starts from scratch."""
    if os.path.isdir(build_dir) and not is_complete(build_dir, module_name):
        shutil.rmtree(build_dir, ignore_errors=True)

#----------------------------------------------------------------------------
# Source staging (atomic, timestamp-stable so ninja rebuilds stay incremental).

def populate_sources(build_dir, source_files):
    if os.path.isdir(build_dir):
        return
    tmpdir = os.path.join(os.path.dirname(build_dir), f'srctmp-{uuid.uuid4().hex}')
    os.makedirs(tmpdir)
    for src in source_files:
        shutil.copyfile(src, os.path.join(tmpdir, os.path.basename(src)))
    try:
        os.replace(tmpdir, build_dir) # atomic
    except OSError:
        # Entry appeared concurrently; discard our copy.
        shutil.rmtree(tmpdir)
        if not os.path.isdir(build_dir):
            raise

#----------------------------------------------------------------------------
# Loading a completed entry (no compiler, no CUDA toolkit, no ninja).

def import_from_cache(module_name, build_dir):
    if module_name in sys.modules:
        return sys.modules[module_name]
    filename = _artifact_path(build_dir, module_name)
    loader = importlib.machinery.ExtensionFileLoader(module_name, filename)
    spec = importlib.util.spec_from_file_location(module_name, filename, loader=loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        loader.exec_module(module)
    except ImportError as err:
        del sys.modules[module_name]
        try:
            with open(os.path.join(build_dir, _COMPLETE_MARKER)) as f:
                provenance = ', '.join(line.strip() for line in f if '=' in line)
        except OSError:
            provenance = 'unknown'
        raise ImportError(f'{err}\nCached op failed to load; it may have been built for a different '
                          f'environment ({provenance}). Ops built with a CUDA toolkit newer than the torch '
                          f'runtime need a correspondingly recent GPU driver. Delete {build_dir} to force '
                          f'a rebuild.') from err
    return module

#----------------------------------------------------------------------------
# Compiler environment.

def _find_compiler_bindir():
    patterns = [
        'C:/Program Files*/Microsoft Visual Studio/*/Professional/VC/Tools/MSVC/*/bin/Hostx64/x64',
        'C:/Program Files*/Microsoft Visual Studio/*/BuildTools/VC/Tools/MSVC/*/bin/Hostx64/x64',
        'C:/Program Files*/Microsoft Visual Studio/*/Community/VC/Tools/MSVC/*/bin/Hostx64/x64',
        'C:/Program Files*/Microsoft Visual Studio */vc/bin',
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if len(matches):
            return matches[-1]
    return None

def setup_compiler_env():
    """Make sure the C++ compiler is reachable; raise if it cannot be found."""
    if os.name == 'nt' and os.system('where cl.exe >nul 2>nul') != 0:
        compiler_bindir = _find_compiler_bindir()
        if compiler_bindir is None:
            raise RuntimeError(f'Could not find MSVC/GCC/CLANG installation on this computer. Check _find_compiler_bindir() in "{__file__}".')
        os.environ['PATH'] += ';' + compiler_bindir

def toolkit_version():
    """Version of the CUDA toolkit that nvcc builds with (e.g. '12.8'),
    or None if no toolkit can be found. This can differ from
    torch.version.cuda, the runtime torch was built against."""
    cuda_home = torch.utils.cpp_extension._find_cuda_home() # pylint: disable=protected-access
    if cuda_home is None:
        return None
    try:
        output = subprocess.check_output([os.path.join(cuda_home, 'bin', 'nvcc'), '--version']).decode()
    except (OSError, subprocess.CalledProcessError):
        return None
    match = re.search(r'release (\d+\.\d+)', output)
    return match.group(1) if match else None

def pin_arch_list(capability=None):
    # Match upstream custom_ops: an empty TORCH_CUDA_ARCH_LIST makes nvcc
    # target the current device, and overriding neutralizes container-set
    # values that could break the build or target the wrong archs. The
    # precompile script passes an explicit capability to cross-build.
    os.environ['TORCH_CUDA_ARCH_LIST'] = capability if capability is not None else ''

def toolkit_mismatch():
    """(toolkit, runtime) versions when their majors differ, else None."""
    toolkit = toolkit_version()
    runtime = torch.version.cuda
    if toolkit is None or runtime is None or toolkit.split('.')[0] == runtime.split('.')[0]:
        return None
    return toolkit, runtime

def warn_toolkit_mismatch():
    """Print the toolkit/runtime mismatch warning once per process."""
    mismatch = toolkit_mismatch()
    if mismatch is not None and not getattr(warn_toolkit_mismatch, '_warned', False):
        warn_toolkit_mismatch._warned = True
        toolkit, runtime = mismatch
        print(f'Warning: building with CUDA toolkit {toolkit} but torch runs CUDA {runtime}. '
              f'This works, but the built ops embed the CUDA {toolkit.split(".")[0]} runtime and need a '
              f'correspondingly recent GPU driver on every machine that loads them. Prefer a CUDA '
              f'{runtime.split(".")[0]}.x toolkit (set CUDA_HOME/CUDA_PATH to select one).', file=sys.stderr)
    return mismatch

#----------------------------------------------------------------------------
