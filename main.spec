# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[
        (r'd:\Anaconda3\envs\autolume\Lib\site-packages\glfw\glfw3.dll', '.'),
        (r'd:\anaconda3\envs\autolume\lib\site-packages\nvidia\cudnn\bin\cudnn64_8.dll', 'nvidia/cudnn/bin'),
        (r'd:\anaconda3\envs\autolume\lib\site-packages\nvidia\cudnn\bin\cudnn64_9.dll', 'nvidia/cudnn/bin'),
        (r'd:\anaconda3\envs\autolume\lib\site-packages\nvidia\cublas\bin\cublas64_11.dll', 'nvidia/cublas/bin'),
        (r'd:\anaconda3\envs\autolume\lib\site-packages\nvidia\cublas\bin\cublasLt64_11.dll', 'nvidia/cublas/bin'),
        (r'd:\anaconda3\envs\autolume\lib\site-packages\nvidia\cublas\bin\cublas64_12.dll', 'nvidia/cublas/bin'),
        (r'd:\anaconda3\envs\autolume\lib\site-packages\nvidia\cublas\bin\cublasLt64_12.dll', 'nvidia/cublas/bin'),
        (r'd:\anaconda3\envs\autolume\lib\site-packages\pywin32_system32\pythoncom310.dll', '.'),
        (r'd:\anaconda3\envs\autolume\lib\site-packages\pywin32_system32\pywintypes310.dll', '.')
    ],
    datas=[
        ('architectures', 'architectures'), ('training', 'training'), 
        (r'd:\anaconda3\envs\autolume\lib\site-packages\clip\bpe_simple_vocab_16e6.txt.gz', 'clip'),
        (r'd:\anaconda3\envs\autolume\lib\site-packages\xformers', 'xformers'),
        (r'd:\anaconda3\envs\autolume\lib\site-packages\tensorrt', 'tensorrt'),
        (r'd:\anaconda3\envs\autolume\lib\site-packages\tensorrt_bindings', 'tensorrt_bindings'),
        (r'd:\anaconda3\envs\autolume\lib\site-packages\tensorrt_libs', 'tensorrt_libs'),
        (r'd:\anaconda3\envs\autolume\lib\site-packages\cuda', 'cuda'),
        (r'd:\anaconda3\envs\autolume\lib\site-packages\win32\win32api.pyd', '.'),
        (r'd:\anaconda3\envs\autolume\lib\site-packages\win32ctypes\pywintypes.py', '.')
    ],
    hiddenimports=['diffusers'],
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
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=r'C:\Users\Metacreation Lab\Desktop\WENHAO\autolumelive_colab\autolume_icon.jpg'
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
