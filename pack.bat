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
call pyinstaller.exe main.spec -y

if %ERRORLEVEL% neq 0 (
    echo "Error: Failed to execute PyInstaller"
    pause
    exit /b
)

echo Copying executables...
xcopy ffprobe.exe dist\main\ /y
xcopy ffmpeg.exe dist\main\ /y
xcopy ninja.exe dist\main\ /y

echo Copying assets and models...
xcopy assets dist\main\assets /s /e /y /i
xcopy sr_models dist\main\sr_models /s /e /y /i
if not exist dist\main\models mkdir dist\main\models
xcopy models\ffhq.pkl dist\main\models\ /y

echo Creating directories...
if not exist dist\main\screenshots mkdir dist\main\screenshots
if not exist dist\main\recordings mkdir dist\main\recordings
if not exist dist\main\training-runs mkdir dist\main\training-runs

echo Packing finished!
pause