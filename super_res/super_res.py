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
from torchvision import transforms
from super_res.net_base import SRVGGNetPlus, SRVGGNetCompact, RRDBNet

def load_model(choice,path):
  if choice =='Quality':
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4).to('cuda')
    model_sd=torch.load(path)['params_ema']
    model.load_state_dict(model_sd)
  if choice =='Balance':
    model = model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu').to('cuda')
    model_sd=torch.load(path)['params']
    model.load_state_dict(model_sd)

  if choice =='Fast':
    model = SRVGGNetPlus(num_in_ch=3, num_out_ch=3, num_feat=48, upscale=4, act_type='prelu').to('cuda')
    model_sd=torch.load(path)
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

class Reader:
    def __init__(self, width, height, video_path):
      self.width=width
      self.height=height
      self.stream_reader = (
                ffmpeg.input(video_path).output('pipe:', format='rawvideo', pix_fmt='bgr24',
                                                loglevel='error').run_async(
                                                    pipe_stdin=True, pipe_stdout=True))
    def get_frame_from_stream(self):
        img_bytes = self.stream_reader.stdout.read(self.width * self.height * 3)  # 3 bytes for one pixel
        if not img_bytes:
            return None
        img = np.frombuffer(img_bytes, np.uint8).reshape([self.height, self.width, 3])
        return img

    def get_frame(self):
        return self.get_frame_from_stream()

class Writer:

    def __init__(self, args, audio, height, width, video_save_path, fps):
        print("SAVING VIDEO TO: ", video_save_path)
        if args.scale_mode:
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
  parser.add_argument("--input_path", type=str, required=True, help="path of input file, mp4")
  parser.add_argument("--model_type", type=str, required=True, choices=['Quality','Balance','Fast'],help="types of model")
  parser.add_argument("--outscale", type=float, default=4, choices=range(1,9), help="scale_factor")
  parser.add_argument("--out_width", type=int, help="output_width")
  parser.add_argument("--out_height", type=int, help="output_height")
  parser.add_argument("--sharpen_scale", type=float, default=2, help="sharpen scale factor")
  parser.add_argument("--scale_mode", type=int, default=0, help="Scaling mode to use 0=custom widthxheight and 1=scale factor")

  return parser


def process(args,file):
  print("Processing", args)
  if args.model_type=="Quality":
    model_path= "../sr_models/Quality.pth"
  elif args.model_type=="Balance":
    model_path= "../sr_models/Balance.pth"
  elif args.model_type=="Fast":
    model_path= "../sr_models/Fast.pt"

  upsampler=load_model(args.model_type,model_path)
  head, tail = os.path.split(file)
  if file[-3:] == 'mp4' or file[-3:] == 'avi' or file[-3:] == 'mov':
    width, height = get_resolution(file)

    if args.outscale > 4 or (check_width_height(args) and (args.out_width > 4*width or args.out_height > 4*height)):
      print('warning: Any super-res scale larger than x4 required non-model inference with interpolation and can be slower')


    audio = get_audio(file)
    if args.scale_mode:
      video_save_path = os.path.join(args.result_path, tail[
                                                         :-4] + f'_result_{args.model_type}_{int(width * args.outscale)}x{int(height * args.outscale)}_Sharpness{args.sharpen_scale}.mp4')
    else:
      video_save_path = os.path.join(args.result_path, tail[
                                                         :-4] + f'_result_{args.model_type}_x{int(args.out_width)}x{int(args.out_height)}_Sharpness{args.sharpen_scale}.mp4')

    cap = cv2.VideoCapture(file)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Framecount",frame_count)
    pbar = tqdm(total=frame_count, unit='frame', desc='inference')

    fps = cap.get(cv2.CAP_PROP_FPS)

    reader= Reader(width,height,file)
    writer = Writer(args, audio, height, width, video_save_path, fps=fps)

    while True:
      img = reader.get_frame()
      if img is not None:
        input=torch.tensor(img).permute(2,0,1).float().to('cuda')/255
        input=torch.unsqueeze(input,0)
        with torch.inference_mode():
          output = upsampler(input)
          output=F.adjust_sharpness(output,args.sharpen_scale)*255

          output = output[0].permute(1,2,0).cpu().numpy().astype(np.uint8)
        
          if args.scale_mode:
            if args.outscale != 4:
              output = cv2.resize(
                output, (
                    int(width * args.outscale),
                    int(height * args.outscale),
                ), interpolation=cv2.INTER_LINEAR)


          else:
            output = cv2.resize(
                output, (
                    int(args.out_width),
                    int(args.out_height),
                ), interpolation=cv2.INTER_LINEAR)

      
        writer.write_frame(output)
        pbar.update(1)
        ret, img = cap.read()

      else:
        print('break')
        break

    writer.close()

  elif file[-3:] == 'jpg' or file[-3:] == 'png':
    data_transformer = transforms.Compose([transforms.ToTensor()])
    image = cv2.imread(file)
    input_width, input_height = image.shape[0], image.shape[1]
    print("INPUT DIMENSIONS", input_width, input_height, image.shape)
    image = data_transformer(image).to('cuda')
    input = torch.unsqueeze(image, 0)

    with torch.inference_mode():
          output = upsampler(input)
          print("OUTPUT DIMENSIONS", output.shape)
          output = F.adjust_sharpness(output, args.sharpen_scale) * 255

          output = output[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
          if args.scale_mode:
              if args.outscale != 4:
                  output = cv2.resize(
                      output, (
                          int(input_width * args.outscale),
                          int(input_width * args.outscale),
                      ), interpolation=cv2.INTER_LINEAR)


          else:
              output = cv2.resize(
                  output, (
                      int(args.out_width),
                      int(args.out_height),
                  ), interpolation=cv2.INTER_LINEAR)

    if args.scale_mode:
      print("USING these params", input_width, input_height, args.outscale)
      path = os.path.join(args.result_path,
                            tail[
                            :-4] + f'_result_{args.model_type}_{int(input_width * args.outscale)}x{int(input_height * args.outscale)}_Sharpness{args.sharpen_scale}.jpg')

    else:
      path = os.path.join(args.result_path,
                          tail[
                          :-4] + f'_result_{args.model_type}_{int(args.out_width)}x{int(args.out_height)}_Sharpness{args.sharpen_scale}.jpg')

    print("Saving image to {}".format(path))
    cv2.imwrite(path, output)


# file loop
def main(args):
  list_file=args.input_path

  #if args output path does not exist
  if not os.path.exists(args.result_path):
    os.makedirs(args.result_path)

  print(list_file)
  for file in list_file:
    print(f'working on {file}')
    if file[-3:] == 'jpg' or file[-3:] == 'png':
      process(args,file)
    if file[-3:] == 'mp4' or file[-3:] == 'avi' or args.input_path[-3:] == 'mov':
      print(f'working on {file}')
      process(args,file)
  print('Done')

if __name__ == '__main__':
    parser = base_args()
    args=parser.parse_args()
    main(args)
