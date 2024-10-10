param(
  [switch]$SkipClone,
  [switch]$CPUOnly
)

# Define python path
$systemPython = "C:\Users\ArthurDeleu\AppData\Local\Programs\Python\Python310\python.exe"

# Define install location
$installLocation = "$env:LOCALAPPDATA\autolumelive_colab"
Write-Host $installLocation

# Go To Install Location -------------------------------------------------------------------------#

if (-not (Test-Path -Path $installLocation -PathType Container)) {
    New-Item -ItemType Directory -Path $installLocation

    if (-not $?)
    {
        throw "Failed to create installation folder."
    }
}

Push-Location $installLocation

# Create Temp Directory --------------------------------------------------------------------------#

$tempDir = [System.IO.Path]::Combine([System.IO.Path]::GetTempPath(), "autolumelive_colab")

if (-not (Test-Path -Path $tempDir -PathType Container)) {
    New-Item -ItemType Directory -Path $tempDir

    if (-not $?)
    {
        throw "Failed to create temp folder."
    }
}

# Install Git ------------------------------------------------------------------------------------#

Write-Host "=> Step: Install Git"

if (-not (Get-Command git -ErrorAction SilentlyContinue))
{
    $gitUrlRepository = "https://api.github.com/repos/git-for-windows/git/releases/latest"
    $gitDownloadInfo = Invoke-RestMethod -Method Get -Uri $gitUrlRepository | % assets | where name -like "*64-bit.exe"
    $gitFilePath = [System.IO.Path]::Combine($tempDir, "git.exe")

    Invoke-WebRequest -Uri $gitDownloadInfo.browser_download_url -OutFile $gitFilePath

    $gitInstallArgs = "/SP- /VERYSILENT /SUPPRESSMSGBOXES /NOCANCEL /NORESTART /CLOSEAPPLICATIONS /RESTARTAPPLICATIONS"
    Start-Process -FilePath $gitFilePath -ArgumentList $gitInstallArgs -Wait

    if (-not $?)
    {
        throw "Failed to install git."
    }
}

# Clone Repository -------------------------------------------------------------------------------#

Write-Host "=> Step: Clone Repository"

if (-not $SkipClone)
{
    if ((Get-ChildItem -Path $installLocation | Measure-Object).Count -eq 0)
    {
        $env:GIT_REDIRECT_STDERR = '2>&1'
        & git clone --depth 1 -b main https://github.com/Metacreation-Lab/autolume.git $installLocation

        if (-not $?)
        {
            throw "Failed to clone the repository."
        }
    }
    else
    {
        $env:GIT_REDIRECT_STDERR = '2>&1'
        & git reset --hard origin/main

        if (-not $?)
        {
            throw "Failed to sync the repository."
        }
    }
}

# Check for ffmpeg.zip and extract ffmpeg.exe -------------------------------------------------#

Write-Host "=> Step: Extract ffmpeg"

    $ffmpegZipPath = Join-Path -Path $installLocation -ChildPath "ffmpeg.zip"
    $ffmpegExtractPath = Join-Path -Path $installLocation -ChildPath "ffmpeg"
    $ffmpegExePath = Join-Path -Path $installLocation -ChildPath "ffmpeg.exe"

    if (Test-Path $ffmpegZipPath) {
        Expand-Archive -Path $ffmpegZipPath -DestinationPath $ffmpegExtractPath
        Copy-Item -Path (Join-Path -Path $ffmpegExtractPath -ChildPath "ffmpeg.exe") -Destination $ffmpegExePath -Force
    }

    Remove-Item $ffmpegZipPath -Force
    Remove-Item $ffmpegExtractPath -Recurse -Force

# Check for ffmpeg.zip and extract ffmpeg.exe -------------------------------------------------#

Write-Host "=> Step: Extract ffprobe"

    $ffprobeZipPath = Join-Path -Path $installLocation -ChildPath "ffprobe.zip"
    $ffprobeExtractPath = Join-Path -Path $installLocation -ChildPath "ffprobe"
    $ffprobeExePath = Join-Path -Path $installLocation -ChildPath "ffprobe.exe"

    if (Test-Path $ffprobeZipPath) {
        Expand-Archive -Path $ffprobeZipPath -DestinationPath $ffmpegExtractPath
        Copy-Item -Path (Join-Path -Path $ffprobeExtractPath -ChildPath "ffprobe.exe") -Destination $ffmpegExePath -Force
    }

    Remove-Item $ffprobeZipPath -Force
    Remove-Item $ffprobeExtractPath -Recurse -Force

# Install Build Tools ----------------------------------------------------------------------------#

Write-Host "=> Step: Install Build Tools"

$buildToolsDownloadURL = "https://aka.ms/vs/17/release/vs_buildtools.exe"
$buildToolsInstallerPath = [System.IO.Path]::Combine($tempDir, "vs_buildtools.exe")
Invoke-WebRequest -Uri $buildToolsDownloadURL -OutFile $buildToolsInstallerPath

if (-not $?)
{
    throw "Failed to download Build Tools."
}

$buildToolsInstallOptions = "--add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.Windows10SDK.19041 --quiet --wait --norestart"
Start-Process -FilePath $buildToolsInstallerPath -ArgumentList $buildToolsInstallOptions -Wait

if (-not $?)
{
    throw "Failed to install Build Tools."
}

# Create Python Virtual Environment --------------------------------------------------------------#

Write-Host "=> Step: Create Python Virtual Environment"

$venvDir = [System.IO.Path]::Combine($installLocation, "venv")
. $systemPython -m venv $venvDir

if (-not $?)
{
    throw "Failed to create Python Virtual Environment."
}

# Copy Python Libs to Virtual Environment --------------------------------------------------------#

$systemPythonDir = [System.IO.Path]::GetDirectoryName($systemPython)
$systemPythonLibsDir = [System.IO.Path]::Combine($systemPythonDir, "libs")
$venvPythonLibsDir = [System.IO.Path]::Combine($venvDir, "Scripts\libs")
Copy-Item -Path $systemPythonLibsDir -Destination $venvPythonLibsDir -Recurse -Force

# Activate Python Virtual Environment ------------------------------------------------------------#

$activatePath = [System.IO.Path]::Combine($venvDir, "Scripts\Activate.ps1")
& $activatePath

if (-not $?)
{
    throw "Failed to activate Python Virtual Environment."
}

# Upgrade pip ------------------------------------------------------------------------------------#

Write-Host "=> Step: Upgrade pip"

. python -m pip install --upgrade pip

if (-not $?)
{
    throw "Failed to upgrade pip."
}

# Install Torch ----------------------------------------------------------------------------------#

Write-Host "=> Step: Install Torch"


if ($CPUOnly)
{
    . python -m pip install torch torchvision torchaudio
}
else
{
    . python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
}

if (-not $?)
{
    throw "Failed to install Torch."
}

# Install Python Requirements --------------------------------------------------------------------#

Write-Host "=> Step: Install Python Requirements"

. python -m pip install -r requirements.txt

# Re-Install ffmpeg-python --------------------------------------------------------------------#

Write-Host "=> Step: Re-Install Ffmpeg-Python"

. python -m pip uninstall -y ffmpeg-python==0.2.0
. python -m pip install ffmpeg-python==0.2.0

# Create Shortcut on Desktop ---------------------------------------------------------------------#

Write-Host "=> Step: Create Shortcut on Desktop"

$runScriptPath = [System.IO.Path]::Combine($installLocation, "run.ps1")
$desktopPath = [System.Environment]::GetFolderPath('Desktop')
$runShortcutPath = [System.IO.Path]::Combine($desktopPath, "AutolumeLiveColab.lnk")

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($runShortcutPath)
$shortcut.TargetPath = "powershell.exe"
$shortcut.Arguments = " -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$runScriptPath`""
$shortcut.Save()

# Run --------------------------------------------------------------------------------------------#

Write-Host "=> Runinng"

. python main.py

#-------------------------------------------------------------------------------------------------#

Remove-Item -LiteralPath $tempDir -Force -Recurse

Read-Host -Prompt "Press Enter to exit..."
