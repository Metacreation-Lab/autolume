# Autolume

Autolume-Live is a tool for real-time visual performances using StyleGAN.

For detailed documentation and demos look at the [Metacreation Lab Website](https://www.metacreation.net/autolume).

![Autolume-live Screen](images/autolume-screen.png)

## Downloads

Latest release:
- [Autolume 2.16.1 for Windows](https://drive.google.com/file/d/1yJHQtIYcVgY9wkj8QG8tIbOxB1btmCtB/view?usp=drive_link)

## Examples

For examples of artworks created with Autolume see: https://www.metacreation.net/artworks

## Development instructions

### Dependencies

- CUDA 12.8 ([download link](https://developer.nvidia.com/cuda-12-8-0-download-archive))
  - Minimum components: CUDA Development + CUDA Runtime
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/download)

#### Windows only

- Microsoft C++ Build Tools ([download link](https://download.visualstudio.microsoft.com/download/pr/13907dbe-8bb3-4cfe-b0ae-147e70f8b2f3/a3193e6e6135ef7f598d6a9e429b010d77260dba33dddbee343a47494b5335a3/vs_BuildTools.exe))
  - Minimum components: Desktop development with C++

#### Linux only (Ubuntu 24.04)

```bash
sudo apt install portaudio19-dev ffmpeg avahi-daemon
sudo systemctl enable --now avahi-daemon
```

Install the [NDI SDK for Linux](https://ndi.video/tools/ndi-sdk/):

```bash
curl -L https://downloads.ndi.tv/SDK/NDI_SDK_Linux/Install_NDI_SDK_v6_Linux.tar.gz -o /tmp/ndi_sdk.tar.gz
tar -xzf /tmp/ndi_sdk.tar.gz -C /tmp
cd /tmp && sh Install_NDI_SDK_v6_Linux.sh
```

The installer creates a directory with spaces in its name. Create a symlink for easier use:

```bash
ln -s "/tmp/NDI SDK for Linux" /tmp/ndi-sdk
```

### Initial setup

Create the Python environment:

```bash
conda create -n autolume python=3.10
conda activate autolume
pip install -r requirements.txt
```

#### Linux: install ndi-python from source

The `ndi-python` pip package does not ship pre-built Linux wheels. Build it from the git repo using the NDI SDK symlink:

```bash
git clone --recursive https://github.com/buresu/ndi-python.git /tmp/ndi-python
export CMAKE_ARGS="-DNDI_SDK_DIR=/tmp/ndi-sdk"
pip install /tmp/ndi-python -v
```

#### Download pre-trained models

Download pre-trained models from [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) and [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch):

```bash
curl -L 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth' -o 'sr_models/Quality.pth'
curl -L 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth' -o 'sr_models/Balance.pth'
mkdir -p training/distillation/Util/face_parsing/pretrained_model
curl -L 'https://drive.google.com/uc?export=download&id=154JgKpzCPW82qINcVieuPH3fZ2e0P812' -o 'training/distillation/Util/face_parsing/pretrained_model/79999_iter.pth'
```

#### Download FFmpeg binaries (Windows only)

```bash
mkdir bin
ffdl download 7.1.1@full -d bin -y
tar -xf bin\ffmpeg-7.1.1-full_build.zip -C bin
```

On Linux, FFmpeg is installed via `apt` (see [Linux dependencies](#linux-only-ubuntu-2404) above).

#### Download default FFHQ model

```bash
mkdir -p models
curl -L 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/research/stylegan2/1/files?redirect=true&path=stylegan2-ffhq-512x512.pkl' -o 'models/stylegan2-ffhq-512x512.pkl'
```

### Running the program

Activate the environment (if not already activated)
```bash
conda activate autolume
```

Start the program
```bash
python main.py
```

### Create release for Windows

Run the release script
```bash
release.bat
```

The release will be output to `dist/Autolume`

### Building documentation

```bash
mkdocs build
```

The documentation will be output to `site`.

To serve the documentation locally:

```bash
mkdocs serve
```

The documentation will be served at `http://127.0.0.1:8000/`.

## Credits

- Philippe Pasquier: Director of Metacreation Lab for Creative AI, Principal Investigator 
- Arshia Sobhan: Project management
- Lionel Ringenbach: Software development
- Michael Tjokrowardojo: Software development
- Jonas Kraasch: Project first-phase development
- Arthur Deleu: Project development
- Mahshid Jabari: DevOps
- Peter Chen: Contributions to the super-resolution module
- Manuel Bolduc: User testing
- Ahmed Abuzuraiq: Research associate

## Acknowledgment

StyleGAN3: https://github.com/NVlabs/stylegan3

GAN compression: https://github.com/lychenyoko/content-aware-gan-compression

GANSpace: https://github.com/harskish/ganspace

## License

Some of the dependencies fall under the [Nvidia Source Code License](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/LICENSE.txt).

## Project status

This code base is still open to changes and bugs might still appear. Please create issues and let us know so we can polish it for the final release!

As a dedicated research team behind Autolume, we emphasize that our system is a tool for exploration and innovation without direct support.
