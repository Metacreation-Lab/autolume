@echo off

E:\develop_tool\anaconda\envs\autolume_github\Scripts\pyinstaller.exe main.spec -y && xcopy assets dist\main\assets /s /e /y /i && xcopy sr_models dist\main\sr_models /s /e /y /i && xcopy models dist\main\models /s /e /y /i&& echo packing finished

pause