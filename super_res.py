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
from net_base import SRVGGNetPlus, SRVGGNetCompact, RRDBNet
from torchvision import transforms

def load_model(choice,path):
  if choice =='Quality':
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    model_sd=torch.load(path)['params_ema']
    model.load_state_dict(model_sd)
  if choice =='Balance':
    model = model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu').to('cuda')
    model_sd=torch.load(path)['params']
    model.load_state_dict(model_sd)

  if choice =='Fast':
    model = SRVGGNetPlus(num_in_ch=3, num_out_ch=3, num_feat=48, upscale=4, act_type='prelu').to('cuda')
    model_sd=torch.load(path).state_dict()
    model.load_state_dict(model_sd)
  return model

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



def base_args():
  parser = argparse.ArgumentParser(description="video_super_resolution")

  parser.add_argument("--result_path", type=str, required=True, help="path of result")
  parser.add_argument("--input_path", type=str, required=True, help="path of input file, img or mp4")
  parser.add_argument("--model_type", type=str, required=True, choices=['Quality','Balance','Fast'],help="types of model")
  parser.add_argument("--model_path", type=str, required=True,help="path of model")
  parser.add_argument("--outscale", type=float, default=1, choices=range(1,9), help="scale_factor")
  parser.add_argument("--out_width", type=int, help="output_width")
  parser.add_argument("--out_height", type=int, help="output_height")
  parser.add_argument("--sharpen_scale", type=float, default=4, help="sharpen scale factor")
  parser.add_argument("--fps", type=int, default=30, help="fps")

  return parser

def main(args):

  upsampler=load_model(args.model_type,args.model_path)

  if args.input_path[-3:]=='mp4' or args.input_path[-3:]=='avi' or args.input_path[-3:]=='mov':

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
    print('done')

  elif args.input_path[-3:]=='jpg' or args.input_path[-3:]=='png':
    data_transformer = transforms.Compose([transforms.ToTensor()])
    image = cv2.imread(args.input_path)
    image = data_transformer(image).to('cuda')
    input=torch.unsqueeze(image,0)

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
      
    cv2.imwrite(args.result_path+'/result.jpg',output)


  
if __name__ == '__main__':
    parser = base_args()
    args=parser.parse_args()
    main(args)