# Autolume-Live

Autolume-Live is a tool for real-time visual performances using StyleGAN2.
For a detailed documentation and demos look at the [Metacreation Website](https://www.metacreation.net/autolume).

## Getting started
### Ubuntu Autolume Installer Guide
To successfully run this program, your system must have a 64-bit Linux operating system, and you must download and install the following prerequisites:
- [Python 3.10](https://www.python.org/downloads/)
- [Git](https://git-scm.com/)

#### Installation Steps

1. Open a terminal.
2. Clone this repository using the following command:
   ```bash
   git clone -b git@github.com:Metacreation-Lab/autolume-live.git
   ```
3. `cd` into the cloned repository
4. Run the installer with `sh installer.sh`

#### Adding ffmpeg

1. Update and upgrade the system packages to get the FFmpeg version currently available in the repository. Execute the following command:
   ```bash
   sudo apt update && sudo apt upgrade
   ```
2. After upgrading the repository, install ffmpeg by running the following command:
   ```bash
   sudo apt install ffmpeg
   ```
3. Confirm that FFmpeg has been installed with:
   ```bash
   ffmpeg -version
   ```

### Windows Autolume Installer Guide

To successfully run this program, your system must have a 64-bit Windows operating system, and you must download and install the following prerequisites:

- **[Autolume Installer](sinstaller.ps1)**
- **[Cuda-Toolkit](https://developer.nvidia.com/cuda-11-7-0-download-archive)**
- **[Python 3.10](https://www.python.org/downloads/release/python-3100/)** (scroll down to files and select a compatible version to your system)

*While installing Python, make sure to select "Add Python to PATH" as shown below:*

![Add Python to PATH](images/python-install.png)

#### Installation Steps

1. Open Command Prompt (`windows+R` > `cmd`)
2. Run the following command: `where python`
   - Ctrl+C the result. It should look like this:
     ```
     C:\Users\Metacreation Lab>where python
     C:\Users\Metacreation Lab\AppData\Local\Programs\Python\Python310\python.exe
     C:\Users\Metacreation Lab\AppData\Local\Microsoft\WindowsApps\python.exe
     ```
   - If more than one result is thrown, copy Python 3.10.

3. Click on the `sinstaller.ps1` file and open it (located in your Downloads folder).
4. Paste the outcome of step 2 to the `systemPython` variable under `#Define python path` in the script:
   ```powershell
   # Define python path
   $systemPython = "C:\Users\Metacreation Lab\AppData\Local\Programs\Python\Python310\python.exe"
   ```

5. If this is the first time running a script in Windows, you have to enable that functionality first. Detailed instructions to do so, can be found **[here](https://medium.com/@dfarkhod/how-to-enable-powershell-script-execution-in-windows-10-49f2b182649d)**

#### Adding ffmpeg

1. Download ffmpeg from [here](https://ffmpeg.org/download.html).
2. Extract the downloaded file.
3. Rename the extracted folder to `ffmpeg`.
4. Type "system variables" into the Windows search bar and click the "Edit the system environment variables" option.
5. Under the "User variables" section, select "Path" and click the "Edit" button.
6. Choose "New" from the side menu.
7. Add `C:\ffmpeg\bin` to the empty field and confirm changes with "OK".

To verify that ffmpeg is correctly added to the Windows PATH, open the Command Prompt or PowerShell and run:
```powershell
ffmpeg
```

Following output confirms the installation:

![FFMPEG installation check](images/ffmpeg-check.png)

8. Go to Downloads and type PowerShell in the address bar and hit Enter.
9. Run the following command inside PowerShell: `.\sinstaller.ps1`
10. Next time you want to run Autolume, simply click the AutolumeLiveColab shortcut on your Desktop.

Place your StyleGAN2 checkpoints (https://github.com/NVlabs/stylegan2-ada-pytorch) in the `models` folder.
Click the AutolumeLiveColab desktop shortcut and get going!

## Examples

Showcase of current state: https://www.dropbox.com/s/e8yfdtdw3x3s7h0/autolume.mp4?dl=0 \
Trial runs:\
    https://www.instagram.com/p/ChsrxkCjJNu/ \
    https://www.instagram.com/p/Cg5aMrqjgIM/ \
Installations: \
    https://metacreation.net/autolume-mzton/ \
    https://metacreation.net/autolume-acedia/ 

## Acknowledgment
PyTorch StyleGAN2-ada: https://github.com/NVlabs/stylegan2-ada-pytorch \
Underlying GUI code: https://github.com/NVlabs/stylegan3 \
GAN compression: https://github.com/lychenyoko/content-aware-gan-compression \
GANSpace: https://github.com/harskish/ganspace

## License
Some of the dependencies fall under the [Nvidia Source Code License](https://github.com/NVlabs/stylegan3/blob/main/LICENSE.txt).

## Project status
This code base is still open to changes and bugs might still appear. Please create issues and let us know so we can polish for the final release!
