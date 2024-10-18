@echo off

E:\develop_tool\anaconda\envs\autolume_gitlab\Scripts\pyinstaller.exe main.spec -y

if %ERRORLEVEL% neq 0 (
    echo "Error: PyInstaller failed!"
    pause
    exit /b
)

xcopy ffprobe.exe dist\main\ /y
xcopy ffmpeg.exe dist\main\ /y
xcopy ninja.exe dist\main\ /y

xcopy assets dist\main\assets /s /e /y /i
xcopy sr_models dist\main\sr_models /s /e /y /i
xcopy models dist\main\models /s /e /y /i

if not exist dist\main\screenshots mkdir dist\main\screenshots
if not exist dist\main\recordings mkdir dist\main\recordings
if not exist dist\main\training-runs mkdir dist\main\training-runs

echo Packing finished!
pause