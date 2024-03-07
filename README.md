# Autolume-Live

Autolume-Live is a tool for real-time visual performances using StyleGAN2.
For a detailed documentation and demos look at the [Metacreation Website](https://www.metacreation.net/autolume).

## Getting started
### Windows Autolume Installer Guide

To successfully run this program, your system must have a 64-bit Windows operating system, and you must download and install the following prerequisites:

1. **[Autolume Installer](sinstaller.ps1)**
2. **[Cuda-Toolkit](https://developer.nvidia.com/cuda-11-7-0-download-archive)**
3. **[Python 3.10](https://www.python.org/downloads/release/python-3100/)** (scroll down to files and select a compatible version to your system)

*While installing Python, make sure to select "Add Python to PATH" as shown below:*

![Add Python to PATH](python-install.png)

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

5. If this is the first time running a script in Windows, you have to enable that functionality first. Detailed instructions to do so, can be found **[here](https://medium.com/@dfarkhod/how-to-enable-powershell-script-execution-in-windows-10-49f2b182649d)**

Steps to add ffmpeg:
Download the ffmpeg from here.
Extract the downloaded file.
Rename the extracted folder to ffmpeg.
Type system variables into the search bar and click the Edit the system environment variables option.
Under the User variables section, select Path and click the Edit button.
Choose New from the side menu.
Add C:\ffmpeg\bin to the empty field and confirm changes with OK.
To verify the FFmpeg is correctly added to the Windows PATH, open the Command Prompt or PowerShell and run: ffmpeg
 The output confirms the installation: 
             
Go to Downloads and type PowerShell in the address bar and hit Enter.
Run the following command inside PowerShell: .\sinstaller.ps1   
Next time you want to run Autolume, simply click the AutolumeLiveColab shortcut on your Desktop.


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

## In Progress
- Make start screenlook nicer
- Training, Compressing, GANspace stop interface and can only be stopped by stopping program
- Sometimes pytorch slows down (seems to aggregate variables in the background), when clearing with gc.collect overall slowdown in render and gui but more stable (currently commented out)
- Super resolution: Off-line super-res module via Real-ESRGAN is now added to GUI and wait for testing. An upgrade model of RRDB net can be implemented for best frame quality. Choices of model is yet to be added for the user to choose different super-res style (photo-realistic or anime-artisitc)


## Acknowledgment
PyTorch StyleGAN2-ada: https://github.com/NVlabs/stylegan2-ada-pytorch \
Underlying GUI code: https://github.com/NVlabs/stylegan3 \
GAN compression: https://github.com/lychenyoko/content-aware-gan-compression \
GANSpace: https://github.com/harskish/ganspace

## License
Some of the dependencies fall under the [Nvidia Source Code License](https://github.com/NVlabs/stylegan3/blob/main/LICENSE.txt).



## Project status
This code base is still open to changes and bugs might still appear. Please create issues and let us know so we can polish for the final release!
