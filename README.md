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

For optimal performance, developers and users must install the following dependencies:
- Microsoft C++ Build Tools ([download link](https://download.visualstudio.microsoft.com/download/pr/13907dbe-8bb3-4cfe-b0ae-147e70f8b2f3/a3193e6e6135ef7f598d6a9e429b010d77260dba33dddbee343a47494b5335a3/vs_BuildTools.exe))
  - Minimum components: Desktop development with C++
- CUDA 12.8 ([download link](https://developer.nvidia.com/cuda-12-8-0-download-archive))
  - Minimum components: CUDA Development + CUDA Runtime

### Initial setup

Install Miniconda from [here](https://docs.conda.io/en/latest/miniconda.html).

Create python environment and install dependencies
```bash
conda create -n autolume python=3.10
conda activate autolume
pip install -r requirements.txt
```

Download pre-trained model from [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) and [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch)

```bash
curl -L 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth' -o 'sr_models/Quality.pth'
curl -L 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth' -o 'sr_models/Balance.pth'
mkdir training/distillation/Util/face_parsing/pretrained_model
curl -L 'https://drive.google.com/uc?export=download&id=154JgKpzCPW82qINcVieuPH3fZ2e0P812' -o 'training/distillation/Util/face_parsing/pretrained_model/79999_iter.pth'
```

Download FFmpeg binaries

```bash
mkdir bin
ffdl download 7.1.1@full -d bin -y
tar -xf bin\ffmpeg-7.1.1-full_build.zip -C bin
```

Download default FFHQ model

```bash
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

## Credits

- Philippe Pasquier: Director of Metacreation Lab for Creative AI, Principal Investigator 
- Arshia Sobhan: Project management
- Lionel Ringenbach: Software development
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
