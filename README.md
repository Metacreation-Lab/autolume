# Autolume-Live

Autolume-Live is a tool for real-time visual performances using StyleGAN2.

## Getting started

Clone the repository and install the packages found in requirements.txt
```
git clone https://gitlab.com/jkraasch/autolume_live.git
cd autolume_live
pip install -r requirements.txt
```
Then place your StyleGAN2 checkpoints (https://github.com/NVlabs/stylegan2-ada-pytorch) in the `models` folder.
Launch the tool by running `python main.py` and get going!

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
