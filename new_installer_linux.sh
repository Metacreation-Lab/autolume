#!/bin/bash

# Define python path
systemPython="/usr/bin/python3"

# Define install location
installLocation="$HOME/.autolumelive_colab"
echo "Install location: $installLocation"

# Go To Install Location
if [ ! -d "$installLocation" ]; then
    mkdir -p "$installLocation" || { echo "Failed to create installation folder."; exit 1; }
fi
cd "$installLocation" || { echo "Failed to change directory to installation folder."; exit 1; }

# Create Temp Directory
tempDir=$(mktemp -d -t autolumelive_colab.XXXXXX) || { echo "Failed to create temp folder."; exit 1; }

# Install Git
echo "=> Step: Install Git"
if ! command -v git &> /dev/null; then
    gitUrlRepository="https://api.github.com/repos/git-for-windows/git/releases/latest"
    gitDownloadInfo=$(curl -s "$gitUrlRepository" | jq '.assets[] | select(.name | test("64-bit.exe"))')
    gitFilePath="$tempDir/git.exe"

    curl -L -o "$gitFilePath" "$(echo $gitDownloadInfo | jq -r '.browser_download_url')"

    gitInstallArgs=("/SP-" "/VERYSILENT" "/SUPPRESSMSGBOXES" "/NOCANCEL" "/NORESTART" "/CLOSEAPPLICATIONS" "/RESTARTAPPLICATIONS")
    wine start /wait "$gitFilePath" "${gitInstallArgs[@]}" || { echo "Failed to install git."; exit 1; }
fi

# Clone Repository
echo "=> Step: Clone Repository"
if [ ! $SkipClone ]; then
    if [ -z "$(ls -A $installLocation)" ]; then
        GIT_REDIRECT_STDERR='2>&1' git clone -b bug_fixing_arthur_stable https://gitlab.com/jkraasch/autolumelive_colab.git $installLocation || { echo "Failed to clone the repository."; exit 1; }
    else
        GIT_REDIRECT_STDERR='2>&1' git reset --hard origin/bug_fixing_arthur_stable || { echo "Failed to sync the repository."; exit 1; }
    fi
fi

# Create Python Virtual Environment
echo "=> Step: Create Python Virtual Environment"
venvDir="$installLocation/venv"
$systemPython -m venv "$venvDir" || { echo "Failed to create Python Virtual Environment."; exit 1; }

# Copy Python Libs to Virtual Environment
# Add your code here

echo "Installation completed successfully."
