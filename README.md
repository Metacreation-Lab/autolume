# Autolume

Autolume-Live is a tool for real-time visual performances using StyleGAN2.

For detailed documentation and demos look at the [Metacreation Lab Website](https://www.metacreation.net/autolume).

![Autolume-live Screen](images/autolume-screen.png)

## Getting started

Current version of Autolume only offers a Windows release:
- [Release 2.15 for Windows](https://drive.google.com/file/d/1SjStN2K2_GZB3ACbz4BQLJ91GWeMYbzm/view?usp=sharing)

### Dependencies

For optimal performance, user must install the following dependencies:
- Microsoft C++ Build Tools ([download link](https://download.visualstudio.microsoft.com/download/pr/13907dbe-8bb3-4cfe-b0ae-147e70f8b2f3/a3193e6e6135ef7f598d6a9e429b010d77260dba33dddbee343a47494b5335a3/vs_BuildTools.exe))
  - Minimum components: Desktop development with C++
- CUDA 12.8 ([download link](https://developer.nvidia.com/cuda-12-8-0-download-archive))
  - Minimum components: CUDA Development + CUDA Runtime

## Development instructions

### One time setup

Install Miniconda from [here](https://docs.conda.io/en/latest/miniconda.html).

Create python environment and install dependencies
```bash
conda env create -f environment.yml
conda activate autolume
pip install -U -r nightly-requirements.txt
```

### Running the program

Activate the environment
```bash
conda activate autolume
```

Start the program
```bash
python main.py
```

## Examples

For examples of artworks created with Autolume see: https://www.metacreation.net/artworks

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
PyTorch StyleGAN2-ada: https://github.com/NVlabs/stylegan2-ada-pytorch \
Underlying GUI code: https://github.com/NVlabs/stylegan3 \
GAN compression: https://github.com/lychenyoko/content-aware-gan-compression \
GANSpace: https://github.com/harskish/ganspace

## License
Some of the dependencies fall under the [Nvidia Source Code License](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/LICENSE.txt).

## Project status
This code base is still open to changes and bugs might still appear. Please create issues and let us know so we can polish it for the final release! \
As a dedicated research team behind Autolume, we emphasize that our system is a tool for exploration and innovation without direct support.
