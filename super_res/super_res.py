import logging
import torch
from tqdm import tqdm
import numpy as np
import torchvision.transforms.functional as F
import os
import argparse
import cv2
from torchvision import transforms
from super_res.net_base import SRVGGNetPlus, SRVGGNetCompact, RRDBNet
from utils.device_utils import get_device
from utils import device_utils, video_io
from utils.resource_paths import resource_path
from utils.user_data import cache_path
from utils.downloads import download_file
import threading
import time
import gc

logger = logging.getLogger(__name__)

# Downloadable Real-ESRGAN weights: display name -> (original filename, url).
# "Fast" is omitted -- it is a small custom model bundled with the app.
SR_WEIGHTS = {
    "Quality": ("RealESRGAN_x4plus.pth",
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"),
    "Balance": ("realesr-general-x4v3.pth",
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"),
    # Weak-denoise sibling of Balance, used only for the dataset upscaler's
    # denoise blend (never listed as a selectable model type).
    "BalanceWDN": ("realesr-general-wdn-x4v3.pth",
                   "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth"),
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
            logger.info("Downloading %s super-resolution weights from %s", model_type, url)
            def progress_cb(done, total):
                pass
        if cancel_event is None:
            cancel_event = threading.Event()
        ok = download_file(url, path, cancel_event, progress_cb)
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
  # Run the forward pass in fp16 on GPU. This roughly halves activation memory,
  # which keeps the Quality model under the VRAM budget instead of spilling to
  # system RAM (a ~13x slowdown on Windows/CUDA). CPU stays fp32 (no fp16 speedup).
  if device.type in ('cuda', 'mps'):
    model = model.half()
  return model


def check_width_height(args):
  return args.out_width is not None and args.out_height is not None


def output_size(args, width, height):
  if args.scale_mode:
    return int(width * args.outscale), int(height * args.outscale)
  assert check_width_height(args)  # width and height should be specified together
  return int(args.out_width), int(args.out_height)


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
  logger.debug("Processing %s with args %s", file, args)
  model_path = ensure_sr_weight(args.model_type)

  upsampler=load_model(args.model_type,model_path)
  head, tail = os.path.split(file)
  if file[-3:] == 'mp4' or file[-3:] == 'avi' or file[-3:] == 'mov':
    info = video_io.probe(file)
    width, height = info.width, info.height

    if args.outscale > 4 or (check_width_height(args) and (args.out_width > 4*width or args.out_height > 4*height)):
      logger.warning('Super-res scale larger than x4 requires non-model inference with interpolation and can be slower')

    out_width, out_height = output_size(args, width, height)
    if args.scale_mode:
      video_save_path = os.path.join(args.result_path, tail[
                                                         :-4] + f'_result_{args.model_type}_{out_width}x{out_height}_Sharpness{args.sharpen_scale}.mp4')
    else:
      video_save_path = os.path.join(args.result_path, tail[
                                                         :-4] + f'_result_{args.model_type}_x{out_width}x{out_height}_Sharpness{args.sharpen_scale}.mp4')

    frame_count = int(info.duration * info.fps)
    logger.debug("Frame count: %d", frame_count)
    pbar = tqdm(total=frame_count, unit='frame', desc='inference', disable=None)

    logger.info("Saving video to %s", video_save_path)
    with video_io.VideoReader(file) as reader, video_io.VideoWriter(
        video_save_path, out_width, out_height, info.fps,
        audio_from=file if info.has_audio else None) as writer:
      for img in reader.frames():
        input=torch.tensor(img).permute(2,0,1).float().to(get_device())/255
        input=torch.unsqueeze(input,0).to(next(upsampler.parameters()).dtype)
        with torch.inference_mode():
          output = upsampler(input).float()
          output=F.adjust_sharpness(output,args.sharpen_scale)*255

          output = output[0].permute(1,2,0).cpu().numpy().astype(np.uint8)

          if args.scale_mode:
            if args.outscale != 4:
              output = cv2.resize(output, (out_width, out_height), interpolation=cv2.INTER_LINEAR)
          else:
            output = cv2.resize(output, (out_width, out_height), interpolation=cv2.INTER_LINEAR)

        writer.write(output)
        pbar.update(1)

  elif file[-3:] == 'jpg' or file[-3:] == 'png':
    data_transformer = transforms.Compose([transforms.ToTensor()])
    image = cv2.imread(file)
    input_width, input_height = image.shape[0], image.shape[1]
    image = data_transformer(image).to(get_device())
    input = torch.unsqueeze(image, 0).to(next(upsampler.parameters()).dtype)

    with torch.inference_mode():
          output = upsampler(input).float()
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
      path = os.path.join(args.result_path,
                            tail[
                            :-4] + f'_result_{args.model_type}_{int(input_width * args.outscale)}x{int(input_height * args.outscale)}_Sharpness{args.sharpen_scale}.jpg')

    else:
      path = os.path.join(args.result_path,
                          tail[
                          :-4] + f'_result_{args.model_type}_{int(args.out_width)}x{int(args.out_height)}_Sharpness{args.sharpen_scale}.jpg')

    logger.info("Saving image to %s", path)
    cv2.imwrite(path, output)


# file loop

def _sr_image(model, args, file, tail, file_idx, reply_queue):
    reply_queue.put([file_idx, 0, 1, -1, False])
    logger.info("Super-res image: %s", file)
    data_transformer = transforms.Compose([transforms.ToTensor()])
    image = cv2.imread(file)
    input_height, input_width = image.shape[0], image.shape[1]
    image = data_transformer(image).to(get_device())
    inp = torch.unsqueeze(image, 0).to(next(model.parameters()).dtype)
    with torch.inference_mode():
        output = model(inp).float()
        output = F.adjust_sharpness(output, args.sharpen_scale) * 255
        output = output[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        if args.scale_mode:
            if args.outscale != 4:
                output = cv2.resize(output, (int(input_width * args.outscale), int(input_height * args.outscale)), interpolation=cv2.INTER_LINEAR)
        else:
            output = cv2.resize(output, (int(args.out_width), int(args.out_height)), interpolation=cv2.INTER_LINEAR)
        path = os.path.join(args.result_path, tail[:-4] + f'_result_{args.model_type}_{int(input_width * args.outscale)}x{int(input_height * args.outscale)}_Sharpness{args.sharpen_scale}.jpg')
        cv2.imwrite(path, output)
    logger.info("Saved %s", path)
    reply_queue.put([file_idx, 1, 1, -1, False])


def _sr_video(model, args, file, tail, file_idx, reply_queue):
    info = video_io.probe(file)
    total_frames = int(info.duration * info.fps)
    out_width, out_height = output_size(args, info.width, info.height)
    video_save_path = os.path.join(args.result_path, tail[:-4] + f'_result_{args.model_type}_{out_width}x{out_height}_Sharpness{args.sharpen_scale}.mp4')
    logger.info("Saving video to %s", video_save_path)
    model_dtype = next(model.parameters()).dtype
    start_time = time.time()
    last_put = start_time
    super_res_idx = 0
    reply_queue.put([file_idx, super_res_idx, total_frames, -1, False])
    with video_io.VideoReader(file) as reader, video_io.VideoWriter(
            video_save_path, out_width, out_height, info.fps,
            audio_from=file if info.has_audio else None) as writer:
        for img in reader.frames():
            with torch.inference_mode():
                sr_input = (torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float().to(get_device()) / 255).to(model_dtype)
                sr_output = model(sr_input).float()
                sr_output = F.adjust_sharpness(sr_output, args.sharpen_scale) * 255
                sr_output = sr_output[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                sr_output = cv2.resize(sr_output, (out_width, out_height), interpolation=cv2.INTER_LINEAR)
                writer.write(sr_output)
            super_res_idx += 1
            now = time.time()
            if now - last_put >= 0.15 or super_res_idx >= total_frames:
                eta = (now - start_time) / super_res_idx * max(total_frames - super_res_idx, 0)
                reply_queue.put([file_idx, super_res_idx, total_frames, eta, False])
                last_put = now


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

  logger.debug("Files to process: %s", list_file)
  for file in list_file:
    logger.info('Super resolution: working on %s', file)
    if file[-3:] == 'jpg' or file[-3:] == 'png':
      process(args,file)
    if file[-3:] == 'mp4' or file[-3:] == 'avi' or args.input_path[-3:] == 'mov':
      process(args,file)
  logger.info('Super resolution done')

if __name__ == '__main__':
    parser = base_args()
    args=parser.parse_args()
    main(args)
