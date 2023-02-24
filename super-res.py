import torch
from torch import nn
from tqdm import tqdm
import numpy as np
from torch.nn import functional as f
import torchvision.transforms.functional as F
import os
import argparse
import ffmpeg
import cv2

class SRVGGNetCompact(nn.Module):
    """A compact VGG-style network structure for super-resolution.
    It is a compact network structure, which performs upsampling in the last layer and no convolution is
    conducted on the HR feature space.
    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_conv (int): Number of convolution layers in the body network. Default: 16.
        upscale (int): Upsampling factor. Default: 4.
        act_type (str): Activation type, options: 'relu', 'prelu', 'leakyrelu'. Default: prelu.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu'):
        super(SRVGGNetCompact, self).__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type

        self.body = nn.ModuleList()
        # the first conv
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        # the first activation
        if act_type == 'relu':
            activation = nn.ReLU(inplace=True)
        elif act_type == 'prelu':
            activation = nn.PReLU(num_parameters=num_feat)
        elif act_type == 'leakyrelu':
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.body.append(activation)

        # the body structure
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            # activation
            if act_type == 'relu':
                activation = nn.ReLU(inplace=True)
            elif act_type == 'prelu':
                activation = nn.PReLU(num_parameters=num_feat)
            elif act_type == 'leakyrelu':
                activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.body.append(activation)

        # the last conv
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        # upsample
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for i in range(0, len(self.body)):
            out = self.body[i](out)

        out = self.upsampler(out)
        # add the nearest upsampled image, so that the network learns the residual
        base = f.interpolate(x, scale_factor=self.upscale, mode='nearest')
        out += base
        return out





def get_resolution(video_path):
  probe = ffmpeg.probe(video_path)
  video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
  w = video_streams[0]['width']
  h = video_streams[0]['height']
  return w,h

def get_audio(video_path):
  probe = ffmpeg.probe(video_path)
  has_audio = any(stream['codec_type'] == 'audio' for stream in probe['streams'])
  audio=ffmpeg.input(video_path).audio if has_audio else None
  return audio

class Writer:

    def __init__(self, args, audio, height, width, video_save_path, fps):
        out_width, out_height = int(width * args.outscale), int(height * args.outscale)

        if audio is not None:
            self.stream_writer = (
                ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{out_width}x{out_height}',
                             framerate=fps).output(
                                 audio,
                                 video_save_path,
                                 pix_fmt='yuv420p',
                                 vcodec='libx264',
                                 loglevel='error',
                                 acodec='copy').overwrite_output().run_async(
                                     pipe_stdin=True, pipe_stdout=True))
        else:
            self.stream_writer = (
                ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{out_width}x{out_height}',
                             framerate=fps).output(
                                 video_save_path, pix_fmt='yuv420p', vcodec='libx264',
                                 loglevel='error').overwrite_output().run_async(
                                     pipe_stdin=True, pipe_stdout=True))

    def write_frame(self, frame):
        frame = frame.astype(np.uint8).tobytes()
        self.stream_writer.stdin.write(frame)

    def close(self):
        self.stream_writer.stdin.close()
        self.stream_writer.wait()


def main():
  parser = argparse.ArgumentParser(description="video_super_resolution")

  parser.add_argument("--result_path", type=str, required=True, help="path of result")
  parser.add_argument("--input_path", type=str, required=True, help="path of input file, mp4")
  parser.add_argument("--model_path", type=str, required=True, help="path of model")
  parser.add_argument("--outscale", type=float, default=4, help="scale_factor")
  parser.add_argument("--sharpen_scale", type=float, default=4, help="sharpen scale factor")
  parser.add_argument("--fps", type=int, default=30, help="fps")

  args = parser.parse_args()

  width, height = get_resolution(args.input_path)
  audio = get_audio(args.input_path)
  video_save_path = os.path.join(args.result_path, f'{args.input_path}_x{args.outscale}.mp4')
  writer = Writer(args, audio, height, width, video_save_path, fps=30)
  outscale=args.outscale

  upsampler=SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu').to('cuda')
  sd=torch.load(args.model_path)['params']
  upsampler.load_state_dict(sd)

  cap = cv2.VideoCapture(args.input_path)
  frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  pbar = tqdm(total=frame_count, unit='frame', desc='inference')
  while True:
    ret, img = cap.read()
    if img is None:
        break
    else:
      input=torch.tensor(img).permute(2,1,0).to('cuda').float()
      input=torch.unsqueeze(input,0)
      with torch.no_grad():
        output = upsampler(input)
        output=F.adjust_sharpness(output,args.sharpen_scale)
        output = output[0].cpu().permute(1,2,0).numpy()
        output_resized = cv2.resize(
                output, (
                    int(width * outscale),
                    int(height * outscale),
                ), interpolation=cv2.INTER_LANCZOS4)

      writer.write_frame(output_resized)

      torch.cuda.synchronize('cuda')
      pbar.update(1)


  writer.close()
  
if __name__ == '__main__':
    main()