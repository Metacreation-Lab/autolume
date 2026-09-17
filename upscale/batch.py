"""Batch upscaling of image and video files for the Upscaling tool.

Runs in its own process so the network gets the GPU at full speed while the
UI stays responsive. Progress is reported back as
``[file_idx, frame_idx, total_frames, eta, done]`` tuples.

The network is a fixed 4x pass, so the output is always 4 times the input
resolution.
"""

import gc
import logging
import os
import time

import cv2
import numpy as np
import torch

from upscale import core
from upscale.core import load_upscaler
from utils import device_utils, video_io

logger = logging.getLogger(__name__)

# The one screen where both models are selectable: Fast for long videos such as
# recorded performances, Quality for stills and short clips.
BATCH_MODELS = ["CompactC3", "RealPLKSR"]
BATCH_LABELS = {"CompactC3": "Fast", "RealPLKSR": "Quality"}
BATCH_DEFAULT_MODEL = "RealPLKSR"

IMAGE_EXTENSIONS = ('jpg', 'jpeg', 'png', 'bmp')
VIDEO_EXTENSIONS = ('mp4', 'avi', 'mov')

# Seconds between progress messages, so the queue is not flooded per frame.
PROGRESS_INTERVAL = 0.15


def output_name(tail, model, width, height, extension):
    """Result filename, carrying the model and size that produced it."""
    stem = os.path.splitext(tail)[0]
    label = BATCH_LABELS.get(model, model)
    return f"{stem}_result_{label}_{width}x{height}{extension}"


def _upscale_frame(model, frame):
    """One fixed 4x network pass."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    inp = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    with torch.inference_mode():
        out = core._tiled_forward(model, inp.to(dtype)).float().clamp_(0.0, 1.0) * 255.0
    return out[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)


def _sr_image(model, args, file, tail, file_idx, reply_queue):
    reply_queue.put([file_idx, 0, 1, -1, False])
    logger.info("Upscaling image: %s", file)
    image = cv2.imread(file)
    output = _upscale_frame(model, image)
    path = os.path.join(args.result_path,
                        output_name(tail, args.model_type, output.shape[1], output.shape[0],
                                    ".jpg"))
    cv2.imwrite(path, output)
    logger.info("Saved %s", path)
    reply_queue.put([file_idx, 1, 1, -1, False])


def _sr_video(model, args, file, tail, file_idx, reply_queue):
    info = video_io.probe(file)
    total_frames = int(info.duration * info.fps)
    out_width, out_height = info.width * 4, info.height * 4
    video_save_path = os.path.join(args.result_path,
                                   output_name(tail, args.model_type, out_width, out_height,
                                               ".mp4"))
    logger.info("Saving video to %s", video_save_path)
    start_time = time.time()
    last_put = start_time
    frame_idx = 0
    reply_queue.put([file_idx, frame_idx, total_frames, -1, False])
    with video_io.VideoReader(file) as reader, video_io.VideoWriter(
            video_save_path, out_width, out_height, info.fps,
            audio_from=file if info.has_audio else None) as writer:
        for img in reader.frames():
            writer.write(_upscale_frame(model, img))
            frame_idx += 1
            now = time.time()
            if now - last_put >= PROGRESS_INTERVAL or frame_idx >= total_frames:
                eta = (now - start_time) / frame_idx * max(total_frames - frame_idx, 0)
                reply_queue.put([file_idx, frame_idx, total_frames, eta, False])
                last_put = now


def run_batch_upscale(queue, reply_queue):
    args = queue.get()
    while not queue.empty():
        args = queue.get()
    model = load_upscaler(args.model_type)
    files = args.input_path
    for file_idx, file in enumerate(files):
        _, tail = os.path.split(file)
        if file.lower().endswith(IMAGE_EXTENSIONS):
            _sr_image(model, args, file, tail, file_idx, reply_queue)
        elif file.lower().endswith(VIDEO_EXTENSIONS):
            _sr_video(model, args, file, tail, file_idx, reply_queue)
        device_utils.empty_cache()
        gc.collect()
    reply_queue.put([len(files), 0, 1, -1, True])
