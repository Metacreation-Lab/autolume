# -*- mode: python ; coding: utf-8 -*-

# this pyinstaller configuration file was adapted from a previous developer
# who created the first version of Autolume.exe embedding the Python runtime
# we are not clear why any of these configuration are included, it just works
# there may be unnecessary specifications in here and we will need to figure out
# a leaner and automated way to execute bundling to deliver the minimum release

import subprocess
import json

def get_conda_env_path(env_name):
    try:
        result = subprocess.run(['conda', 'info', '--envs', '--json'], 
                              capture_output=True, text=True, check=True)
        conda_info = json.loads(result.stdout)
        for env_path in conda_info['envs']:
            if env_path.endswith(f'\\{env_name}') or env_path.endswith(f'/{env_name}'):
                return env_path
        
        raise RuntimeError(f"Conda environment '{env_name}' not found")
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to get conda info: {e}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse conda info: {e}")

conda_env = get_conda_env_path('autolume')

a = Analysis(
    ['main.py'],
    pathex=[rf'{conda_env}\Lib\site-packages',
            rf'{conda_env}\include',
            r'torch_utils',
            ],
    binaries=[
        (rf'{conda_env}\Lib\site-packages\glfw\glfw3.dll', '.'),
        (rf'{conda_env}\Lib\site-packages\torch\lib\cudnn64_9.dll', 'torch/lib'),
        (rf'{conda_env}\Lib\site-packages\torch\lib\cublas64_12.dll', 'torch/lib'),
        (rf'{conda_env}\Lib\site-packages\torch\lib\cublasLt64_12.dll', 'torch/lib'),
        (rf'{conda_env}\Lib\site-packages\torch\lib\torch.lib', 'torch/lib'),
        (rf'{conda_env}\Lib\site-packages\torch\lib\torch_python.lib', 'torch/lib'),
        (rf'{conda_env}\Lib\site-packages\torch\lib\c10.lib', 'torch/lib'),
        (rf'{conda_env}\Lib\site-packages\torch\lib\c10_cuda.lib', 'torch/lib'),
        (rf'{conda_env}\Lib\site-packages\torch\lib\torch_cpu.lib', 'torch/lib'),
        (rf'{conda_env}\Lib\site-packages\torch\lib\torch_cuda.lib', 'torch/lib'),
        (rf'{conda_env}\Lib\site-packages\ninja\data\bin\ninja.exe', '.'),
        (rf'{conda_env}\Scripts\ninja.exe', '.'),
        (rf'{conda_env}\Library\bin\ffmpeg.exe', '.'),
        (rf'{conda_env}\Library\bin\ffprobe.exe', '.'),
        (rf'{conda_env}\libs\python310.lib', '.'),
    ],
    datas=[
        ('architectures', 'architectures'),
        ('assets', 'assets'),
        ('training', 'training'),
        ('torch_utils', 'torch_utils'),
        (rf'{conda_env}\Lib\site-packages\clip\bpe_simple_vocab_16e6.txt.gz', 'clip'),
        (rf'{conda_env}\Lib\site-packages\torch\cuda', 'cuda'),
        (rf'{conda_env}\Lib\site-packages\win32ctypes\pywintypes.py', '.'),
        (rf'{conda_env}\Lib\site-packages\torch\include', 'torch/include'),
        (rf'{conda_env}\include', 'include'),
        (rf'{conda_env}\Lib\site-packages\torch\include\torch', 'torch/include/torch'),
        (rf'{conda_env}\Lib\site-packages\torch\include\c10', 'torch/include/c10'),
        (rf'{conda_env}\Lib\site-packages\torch\include\ATen', 'torch/include/ATen'),
        (rf'{conda_env}\libs\python310.lib', '.'),
        ('recordings', 'recordings'),
    ],
    hiddenimports=[
        'torch',
        'skimage', 
        'scipy', 
        'torch',  
        'ffmpeg',
        'imageio_ffmpeg', 
        'clip', 
        'kmeans_pytorch', 
        'kornia', 
        'librosa', 
        'numba', 
        'numpy', 
        'cv2', 
        'sklearn',
        'skimage.metrics',
        'lpips',
        'pandas',
        'openpyxl',
        'et_xmlfile',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Autolume',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='main',
)

