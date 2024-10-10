@ECHO OFF

@REM SET PYTHON=
@REM SET GIT=

GOTO :entry_point


:entry_point
    IF NOT DEFINED PYTHON (SET PYTHON=python)
    IF NOT DEFINED GIT (SET GIT=git)
    IF NOT DEFINED VENV_DIR (SET "VENV_DIR=%~dp0%venv")

    SET ERROR_REPORTING=FALSE
    MKDIR tmp 2>NUL

    python --version 2>&1 | FINDSTR /R /C:"^Python 3.10" >NUL
    IF %ERRORLEVEL% EQU 0 (
        GOTO :check_requirements
    )
    ECHO Couldn't launch python. Please install python 3.10: https://www.python.org/downloads/release/python-3100/
    GOTO :endofscript

:check_requirements
    GOTO :check_winsdk_installed
    GOTO :check_pip


:check_winsdk_installed:
    reg query "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows Kits\Installed Roots" >NUL
    IF NOT %ERRORLEVEL% EQU 0 (
        ECHO Windows Software Development Kit is required: https://developer.microsoft.com/en-us/windows/downloads/windows-sdk/
        ECHO Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/
        GOTO :endofscript
    )

:check_pip
    %PYTHON% -mpip --help >tmp/stdout.txt 2>tmp/stderr.txt
    IF %ERRORLEVEL% == 0 (
        GOTO :start_venv
    )
    IF "%PIP_INSTALLER_LOCATION%" == "" (
        GOTO :show_stdout_stderr
    )
    %PYTHON% "%PIP_INSTALLER_LOCATION%" >tmp/stdout.txt 2>tmp/stderr.txt
    if %ERRORLEVEL% == 0 (
        GOTO :start_venv
    )
    ECHO Couldn't install pip
    GOTO :show_stdout_stderr


:start_venv
    IF ["%VENV_DIR%"] == ["-"] (
        GOTO :skip_venv
    )
    IF ["%SKIP_VENV%"] == ["1"] (
        GOTO :skip_venv
    )

    DIR "%VENV_DIR%\Scripts\Python.exe" >tmp/stdout.txt 2>tmp/stderr.txt
    IF %ERRORLEVEL% == 0 (
        GOTO :activate_venv
    )

    FOR /f "delims=" %%i IN (
        'CALL %PYTHON% -c "import sys; print(sys.executable)"'
    ) DO SET PYTHON_FULLNAME="%%i"

    ECHO Creating venv in directory %VENV_DIR% using python %PYTHON_FULLNAME%
    %PYTHON_FULLNAME% -m venv "%VENV_DIR%" >tmp/stdout.txt 2>tmp/stderr.txt

    IF %ERRORLEVEL% == 0 (
        GOTO :activate_venv 
    )

    ECHO Unable to create venv in directory "%VENV_DIR%"
    GOTO :show_stdout_stderr


:activate_venv
    SET PYTHON="%VENV_DIR%\Scripts\Python.exe"
    ECHO venv %PYTHON%
    GOTO :install_requirements


:skip_venv
    GOTO :install_requirements


:install_requirements
    @REM %PYTHON% -mpip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
    %PYTHON% -mpip install -r requirements.txt
    IF %ERRORLEVEL% == 0 (
        GOTO :launch
    )


:launch
    %PYTHON% main.py %*
    PAUSE
    EXIT /b


:show_stdout_stderr
    ECHO.
    ECHO exit code: %ERRORLEVEL%

    FOR /f %%i IN (
        "tmp\stdout.txt"
    ) DO SET size=%%~zi
    IF %size% EQU 0 (
        GOTO :show_stderr
    )
    ECHO.
    ECHO stdout:
    TYPE tmp\stdout.txt


:show_stderr
    FOR /f %%i IN (
        "tmp\stderr.txt"
    ) DO SET size=%%~zi
    IF %size% EQU 0 (
        GOTO :show_stderr
    )
    ECHO.
    ECHO stderr:
    TYPE tmp\stderr.txt


:endofscript
    ECHO.
    ECHO Launch unsuccessful. Exiting.
    PAUSE