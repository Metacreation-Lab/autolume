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
from utils.device_utils import get_device
from utils import device_utils
from utils.resource_paths import resource_path
from utils.user_data import cache_path
from utils.downloads import download_file
import threading
import time
import gc

# Downloadable Real-ESRGAN weights: display name -> (original filename, url).
# "Fast" is omitted -- it is a small custom model bundled with the app.
SR_WEIGHTS = {
    "Quality": ("RealESRGAN_x4plus.pth",
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"),
    "Balance": ("realesr-general-x4v3.pth",
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"),
}


def sr_weight_path(model_type):
    """Resolve the on-disk path for a super-res weight.

    Fast loads from the bundle; Quality/Balance resolve to the user cache dir
    (they may not exist yet -- see :func:`ensure_sr_weight`).
    """
    if model_type in SR_WEIGHTS:
        filename, _ = SR_WEIGHTS[model_type]
        return str(cache_path("real-esrgan", filename))
    return str(resource_path("sr_models", "Fast.pt"))


def ensure_sr_weight(model_type, progress_cb=None, cancel_event=None):
    """Return the weight path, downloading Quality/Balance into the cache if missing.

    Headless-safe: with no progress_cb it prints progress, so CLI callers work.
    Returns None if the download was cancelled.
    """
    path = sr_weight_path(model_type)
    if model_type in SR_WEIGHTS and not os.path.exists(path):
        _, url = SR_WEIGHTS[model_type]
        if progress_cb is None:
            def progress_cb(done, total):
                pct = f"{done / total:6.1%}" if total else f"{done} bytes"
                print(f"\rDownloading {model_type} weights: {pct}", end="", flush=True)
        if cancel_event is None:
            cancel_event = threading.Event()
        ok = download_file(url, path, cancel_event, progress_cb)
        print()
        if not ok:
            return None
    return path


def load_model(choice,path):
  device = get_device()
  if choice =='Quality':
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4).to(device)
    model_sd=torch.load(path, map_location=device)['params_ema']
    model.load_state_dict(model_sd)
  if choice =='Balance':
    model = model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu').to(device)
    model_sd=torch.load(path, map_location=device)['params']
    model.load_state_dict(model_sd)

  if choice =='Fast':
    model = SRVGGNetPlus(num_in_ch=3, num_out_ch=3, num_feat=48, upscale=4, act_type='prelu').to(device)
    model_sd=torch.load(path, map_location=device)
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
  model_path = ensure_sr_weight(args.model_type)

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
        input=torch.tensor(img).permute(2,0,1).float().to(get_device())/255
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
    image = data_transformer(image).to(get_device())
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

def _sr_image(model, args, file, tail, file_idx, reply_queue):
    reply_queue.put([file_idx, 0, 1, -1, False])
    print(f"Super-res image: {file}")
    data_transformer = transforms.Compose([transforms.ToTensor()])
    image = cv2.imread(file)
    input_height, input_width = image.shape[0], image.shape[1]
    image = data_transformer(image).to(get_device())
    inp = torch.unsqueeze(image, 0)
    with torch.inference_mode():
        output = model(inp)
        output = F.adjust_sharpness(output, args.sharpen_scale) * 255
        output = output[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        if args.scale_mode:
            if args.outscale != 4:
                output = cv2.resize(output, (int(input_width * args.outscale), int(input_height * args.outscale)), interpolation=cv2.INTER_LINEAR)
        else:
            output = cv2.resize(output, (int(args.out_width), int(args.out_height)), interpolation=cv2.INTER_LINEAR)
        path = os.path.join(args.result_path, tail[:-4] + f'_result_{args.model_type}_{int(input_width * args.outscale)}x{int(input_height * args.outscale)}_Sharpness{args.sharpen_scale}.jpg')
        cv2.imwrite(path, output)
    print(f"Saved {path}")
    reply_queue.put([file_idx, 1, 1, -1, False])


def _sr_video(model, args, file, tail, file_idx, reply_queue):
    audio = get_audio(file)
    video = cv2.VideoCapture(file)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()
    if args.scale_mode:
        video_save_path = os.path.join(args.result_path, tail[:-4] + f'_result_{args.model_type}_{int(video_width * args.outscale)}x{int(video_height * args.outscale)}_Sharpness{args.sharpen_scale}.mp4')
    else:
        video_save_path = os.path.join(args.result_path, tail[:-4] + f'_result_{args.model_type}_{int(args.out_width)}x{int(args.out_height)}_Sharpness{args.sharpen_scale}.mp4')
    print(f"Saving video to {video_save_path}")
    writer = Writer(args, audio, video_height, video_width, video_save_path=video_save_path, fps=fps)
    reader = Reader(video_width, video_height, file)
    start_time = time.time()
    last_put = start_time
    super_res_idx = 0
    reply_queue.put([file_idx, super_res_idx, total_frames, -1, False])
    while True:
        img = reader.get_frame()
        if img is None:
            break
        with torch.inference_mode():
            sr_input = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float().to(get_device()) / 255
            sr_output = model(sr_input)
            sr_output = F.adjust_sharpness(sr_output, args.sharpen_scale) * 255
            sr_output = sr_output[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            if args.scale_mode:
                sr_output = cv2.resize(sr_output, (int(video_width * args.outscale), int(video_height * args.outscale)), interpolation=cv2.INTER_LINEAR)
            else:
                sr_output = cv2.resize(sr_output, (int(args.out_width), int(args.out_height)), interpolation=cv2.INTER_LINEAR)
            writer.write_frame(sr_output)
        super_res_idx += 1
        print(f"Processing frame {super_res_idx}/{total_frames}")
        now = time.time()
        if now - last_put >= 0.15 or super_res_idx >= total_frames:
            eta = (now - start_time) / super_res_idx * max(total_frames - super_res_idx, 0)
            reply_queue.put([file_idx, super_res_idx, total_frames, eta, False])
            last_put = now
    writer.close()


def run_super_res(queue, reply_queue):
    args = queue.get()
    while not queue.empty():
        args = queue.get()
    model_path = sr_weight_path(args.model_type)
    model = load_model(args.model_type, model_path)
    files = args.input_path
    for file_idx, file in enumerate(files):
        head, tail = os.path.split(file)
        if file.lower().endswith(('jpg', 'png', 'jpeg', 'bmp')):
            _sr_image(model, args, file, tail, file_idx, reply_queue)
        elif file.lower().endswith(('mp4', 'avi', 'mov')):
            _sr_video(model, args, file, tail, file_idx, reply_queue)
        device_utils.empty_cache()
        gc.collect()
    reply_queue.put([len(files), 0, 1, -1, True])


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
