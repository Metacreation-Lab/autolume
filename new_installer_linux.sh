param(
  [switch]$SkipClone,
  [switch]$CPUOnly
)

# Define python path
$systemPython = "/usr/bin/python3"

# Define install location
$installLocation = "$HOME/.autolumelive_colab"
echo $installLocation

# Go To Install Location -------------------------------------------------------------------------#

if (-not (Test-Path -Path $installLocation -PathType Container)) {
    mkdir -p $installLocation

    if [ $? -ne 0 ]; then
        echo "Failed to create installation folder."
        exit 1
    fi
}

pushd $installLocation

# Create Temp Directory --------------------------------------------------------------------------#

tempDir=$(mktemp -d -t autolumelive_colab.XXXXXX)

if [ ! -d "$tempDir" ]; then
    echo "Failed to create temp folder."
    exit 1
fi

# Install Git ------------------------------------------------------------------------------------#

echo "=> Step: Install Git"

if ! command -v git &> /dev/null; then
    gitUrlRepository="https://api.github.com/repos/git-for-windows/git/releases/latest"
    gitDownloadInfo=$(curl -s $gitUrlRepository | jq '.assets[] | select(.name | test("64-bit.exe"))')
    gitFilePath="$tempDir/git.exe"

    curl -L -o $gitFilePath $(echo $gitDownloadInfo | jq -r '.browser_download_url')

    gitInstallArgs=("/SP-" "/VERYSILENT" "/SUPPRESSMSGBOXES" "/NOCANCEL" "/NORESTART" "/CLOSEAPPLICATIONS" "/RESTARTAPPLICATIONS")
    wine start /wait $gitFilePath ${gitInstallArgs[@]}

    if [ $? -ne 0 ]; then
        echo "Failed to install git."
        exit 1
    fi
fi

# Clone Repository -------------------------------------------------------------------------------#

echo "=> Step: Clone Repository"

if [ ! $SkipClone ]; then
    if [ $(ls -A $installLocation | wc -l) -eq 0 ]; then
        GIT_REDIRECT_STDERR='2>&1' git clone -b bug_fixing_arthur_stable https://gitlab.com/jkraasch/autolumelive_colab.git $installLocation

        if [ $? -ne 0 ]; then
            echo "Failed to clone the repository."
            exit 1
        fi
    else
        GIT_REDIRECT_STDERR='2>&1' git reset --hard origin/bug_fixing_arthur_stable

        if [ $? -ne 0 ]; then
            echo "Failed to sync the repository."
            exit 1
        fi
    fi
fi

# Install Build Tools ----------------------------------------------------------------------------#

echo "=> Step: Install Build Tools"

buildToolsDownloadURL="https://aka.ms/vs/17/release/vs_buildtools.exe"
buildToolsInstallerPath="$tempDir/vs_buildtools.exe"
curl -L -o $buildToolsInstallerPath $buildToolsDownloadURL

if [ $? -ne 0 ]; then
    echo "Failed to download Build Tools."
    exit 1
fi

buildToolsInstallOptions="--add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.Windows10SDK.19041 --quiet --wait --norestart"
wine start /wait $buildToolsInstallerPath $buildToolsInstallOptions

if [ $? -ne 0 ]; then
    echo "Failed to install Build Tools."
    exit 1
fi

# Create Python Virtual Environment --------------------------------------------------------------#

echo "=> Step: Create Python Virtual Environment"

venvDir="$installLocation/venv"
$systemPython -m venv $venvDir

if [ $? -ne 0 ]; then
    echo "Failed to create Python Virtual Environment."
    exit 1
fi

