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


def check_width_height(args):
  return args.out_width is not None and args.out_height is not None


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
        if args.out_width is None and args.out_height is None:
          out_width, out_height = int(width * args.outscale), int(height * args.outscale)

        else: 
          assert (args.out_width is not None and args.out_height is not None) # width and height should be specify together

          out_width, out_height = int(args.out_width), int(args.out_width)

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
                                     pipe_stdin=True, pipe_stdout=True,cmd='ffmpeg'))
        else:
            self.stream_writer = (
                ffmpeg.input('pipe:', format='rawvideo', 
                pix_fmt='bgr24', 
                s=f'{out_width}x{out_height}',
                             framerate=fps).output(
                                 video_save_path, pix_fmt='yuv420p',vcodec='libx264',
                                 loglevel='error').overwrite_output().run_async(
                                     pipe_stdin=True, pipe_stdout=True,cmd='ffmpeg'))

    def write_frame(self, frame):
        frame = frame.tobytes()
        self.stream_writer.stdin.write(frame)

    def close(self):
        self.stream_writer.stdin.close()
        self.stream_writer.wait()



def main():
  parser = argparse.ArgumentParser(description="video_super_resolution")

  parser.add_argument("--result_path", type=str, required=True, help="path of result")
  parser.add_argument("--input_path", type=str, required=True, help="path of input file, mp4")
  parser.add_argument("--model_path", type=str, required=True, help="path of model")
  parser.add_argument("--outscale", type=float, choices=range(1,9), help="scale_factor")
  parser.add_argument("--out_width", type=int, help="output_width")
  parser.add_argument("--out_height", type=int, help="output_height")
  parser.add_argument("--sharpen_scale", type=float, default=4, help="sharpen scale factor")
  parser.add_argument("--fps", type=int, default=30, help="fps")

  args = parser.parse_args()



  width, height = get_resolution(args.input_path)

  if args.outscale > 4 or (check_width_height(args) and (args.out_width > 4*width or args.out_height > 4*height)):
    print('warning: Any super-res scale larger than x4 required non-model inference with interpolation and can be slower')


  audio = get_audio(args.input_path)
  if check_width_height(args):
    video_save_path = os.path.join(args.result_path, f'result_x{int(args.out_width)}x{int(args.out_height)}.mp4')
  else: 
    video_save_path = os.path.join(args.result_path, f'result_{int(width*args.outscale)}x{int(height*args.outscale)}.mp4')

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
      print('break')
      break
    else:
      input=torch.tensor(img).permute(2,0,1).float().to('cuda')/255
      input=torch.unsqueeze(input,0)
      with torch.inference_mode():
        output = upsampler(input)
        output=F.adjust_sharpness(output,args.sharpen_scale)*255 

        output = output[0].permute(1,2,0).cpu().numpy().astype(np.uint8) 
        
        if check_width_height(args):
          output = cv2.resize( 
                output, (
                    int(args.out_width),
                    int(args.out_height),
                ), interpolation=cv2.INTER_LINEAR)

        else:
          if args.outscale != 4:
            output = cv2.resize( 
                output, (
                    int(width * outscale),
                    int(height * outscale),
                ), interpolation=cv2.INTER_LINEAR)
      
      writer.write_frame(output)
      pbar.update(1)
      ret, img = cap.read()


  writer.close()
  
if __name__ == '__main__':
    main()