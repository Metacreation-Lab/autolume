@echo off

echo Cleaning up old builds...
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build

echo Activating conda environment...
call conda activate autolume

if %ERRORLEVEL% neq 0 (
    echo "Error: Failed to activate conda environment"
    pause
    exit /b
)

echo Running PyInstaller...
call pyinstaller main.py ^
  --name Autolume ^
  --add-binary "bin\ffmpeg-7.1.1-full_build\bin\ffmpeg.exe;." ^
  --add-binary "bin\ffmpeg-7.1.1-full_build\bin\ffprobe.exe;." ^
  --add-binary "%CONDA_PREFIX%\Lib\site-packages\ninja\data\bin\ninja.exe;." ^
  --add-binary "%CONDA_PREFIX%\Lib\site-packages\glfw\glfw3.dll;." ^
  --add-binary "%CONDA_PREFIX%\Lib\site-packages\torch\lib\c10.lib;torch/lib" ^
  --add-binary "%CONDA_PREFIX%\Lib\site-packages\torch\lib\c10_cuda.lib;torch/lib" ^
  --add-binary "%CONDA_PREFIX%\Lib\site-packages\torch\lib\torch_cpu.lib;torch/lib" ^
  --add-binary "%CONDA_PREFIX%\Lib\site-packages\torch\lib\torch_cuda.lib;torch/lib" ^
  --add-binary "%CONDA_PREFIX%\Lib\site-packages\torch\lib\torch.lib;torch/lib" ^
  --add-binary "%CONDA_PREFIX%\Lib\site-packages\torch\lib\torch_python.lib;torch/lib" ^
  --add-binary "%CONDA_PREFIX%\libs\python310.lib;libs" ^
  --add-data "architectures;architectures" ^
  --add-data "assets;assets" ^
  --add-data "training;training" ^
  --add-data "torch_utils;torch_utils" ^
  --add-data "recordings;recordings" ^
  --add-data "%CONDA_PREFIX%\Lib\site-packages\clip\bpe_simple_vocab_16e6.txt.gz;clip" ^
  --add-data "%CONDA_PREFIX%\Lib\site-packages\torch\include;torch/include" ^
  --add-data "%CONDA_PREFIX%\include;include" ^
  --collect-all "lpips"

if %ERRORLEVEL% neq 0 (
    echo Error: Failed to execute PyInstaller
    pause
    exit /b
)

echo Copying assets and models...
xcopy assets dist\Autolume\assets /s /e /y /i
xcopy sr_models dist\Autolume\sr_models /s /e /y /i
if not exist dist\Autolume\models mkdir dist\Autolume\models
<<<<<<< HEAD
xcopy models\stylegan2-ffhq-512x512.pkl dist\Autolume\models\ /y
=======
xcopy models\ffhq.pkl dist\Autolume\models\ /y
>>>>>>> 9775bae (Initial commit with code only (no model files))

echo Creating directories...
if not exist dist\Autolume\screenshots mkdir dist\Autolume\screenshots
if not exist dist\Autolume\recordings mkdir dist\Autolume\recordings
if not exist dist\Autolume\training-runs mkdir dist\Autolume\training-runs

echo Release created in dist/Autolume directory
pause