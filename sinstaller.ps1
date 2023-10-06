# Define git path
$git = "C:\Program Files\Git\cmd\git.exe"

# Define python path
$systemPython = "C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe"

#-------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------#

# Define source path
$srcDir = (Get-Location).Path

# Define Python Virtual Environment path
$venvDir = Join-Path $srcDir "venv"

# Define torch verison
$cudaCommand = ""

#-------------------------------------------------------------------------------------------------#

# Define temp path
$tmpDir = Join-Path $srcDir "tmp"

if (-not (Test-Path -Path $tmpDir -PathType Container))
{
    New-Item -Path $tmpDir -ItemType Directory
}

# Define the URL to vs_buildtools.exe
$buildToolsDownloadURL = "https://aka.ms/vs/17/release/vs_buildtools.exe"

# Specify the path to save vs_buildtools.exe
$buildToolsInstallerPath = Join-Path $tmpDir "vs_buildtools.exe"

Write-Host "Downloading Build Tools"

# Download vs_buildtools.exe
Invoke-WebRequest -Uri $buildToolsDownloadURL -OutFile $buildToolsInstallerPath

if ($?)
{
    Write-Host "Download completed successfully."

    Write-Host "Installing Visual C++ Build Tools..."

    # Specify the installation options (modify as needed)
    $installOptions = "--add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.Windows10SDK.19041 --quiet --wait --norestart"

    # Install Visual C++ Build Tools
    Start-Process -FilePath $buildToolsInstallerPath -ArgumentList $installOptions -Wait

    if ($?)
    {
        Write-Host "Installed Visual C++ Build Tools successfully."


        Write-Host "Creating Python Virtual Environment..."

        # Create Python Virtual Environment
        . $systemPython -m venv $venvDir

        if ($?)
        {
            Write-Host "Created Python Virtual Environment successfully."


            Write-Host "Activing Python Virtual Environment..."

            $activatePath = Join-Path $venvDir "Scripts\Activate.ps1"
            & $activatePath
            
            if ($?)
            {
                Write-Host "Activated Python Virtual Environment successfully."

                Write-Host "Upgrading pip..."

                . python -m pip install --upgrade pip

                if ($?)
                {
                    Write-Host "Upgraded pip successfully."
                    
                    Write-Host "Installing Torch..."
                    . python -m pip install torch torchvision torchaudio

                    if ($?)
                    {
                        Write-Host "Installed Torch successfully."


                        Write-Host "Installing Requirements..."
                        . python -m pip install -r requirements.txt

                        if ($?)
                        {
                            Write-Host "Installed Requirements successfully."

                            . python main.py
                        }
                    }
                    else
                    {
                        Write-Host "Torch Installation failed."
                    }
                }
                else
                {
                    Write-Host "pip Upgrading failed."
                }
            }
            else
            {
                Write-Host "Activing Python Virtual Environment failed."
            }
        }
        else
        {
            Write-Host "Creation Python Virtual Environment failed."
        }
    }
    else
    {
        Write-Host "Installation failed. Please check the error messages above."
    }
} else
{
    Write-Host "Download failed. Please check the error messages above."
}

Read-Host -Prompt "Press Enter to exit..."