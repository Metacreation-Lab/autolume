param(
  [switch]$SkipClone,
  [switch]$CPUOnly
)

# Define python path
$systemPython = "C:\Users\Metacreation Lab\AppData\Local\Programs\Python\Python310\python.exe"

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
        git clone -b windows-installer https://gitlab.com/jkraasch/autolumelive_colab.git $installLocation

        if (-not $?)
        {
            throw "Failed to clone the repository."
        }
    }
}

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