import gc
import multiprocessing
import torch
from torchvision.io import read_video
import dnnlib
from widgets import pipeline
from tqdm import tqdm
from PIL import Image
import numpy as np


def compare_args(args, cur_args):
    if args is None or cur_args is None:
        return False
    for key in args.keys():
        if key == "prompt":  # Skip the prompt parameter
            continue
        if key not in cur_args:
            return False
        a1 = args[key]
        a2 = cur_args.get(key)
        if not isinstance(a1, type(a2)):
            return False
        if isinstance(a1, dict):
            if not compare_args(a1, a2):
                return False
        else:
            if a1 != a2:
                return False
    return True


class DiffusionRender:
    def __init__(self):
        self._closed = False
        self._cur_args = None
        self._cur_result = None
        self._cur_stamp = 0
        self._args_queue = multiprocessing.Queue()
        self._frame_queue = multiprocessing.Queue()
        self._result_queue = multiprocessing.Queue()
        self._process = multiprocessing.Process(target=self._process_fn,
                                                args=(self._args_queue, self._frame_queue, self._result_queue),
                                                daemon=True)
        self._process.start()

    def close(self):
        self._closed = True
        self._renderer_obj = None
        if self._process is not None:
            self._process.terminate()
        self._process = None
        self._args_queue = None
        self._result_queue = None

    @property
    def is_async(self):
        return self._is_async

    def set_async(self, is_async):
        self._is_async = is_async

    def set_args(self, **args):
        if not self._closed:
            if self._args_queue.qsize() == 0:
                self._args_queue.put([args, self._cur_stamp])

    def get_result(self):
        if not self._closed:
            if self._result_queue is not None:
                if self._result_queue.qsize() > 0:
                    result, stamp = self._result_queue.get()
                    while self._result_queue.qsize() > 0:
                        result, stamp = self._result_queue.get()
                    self._cur_result = result
            return self._cur_result

    def clear_result(self):
        assert not self._closed
        self._cur_args = None
        self._cur_result = None
        self._cur_stamp += 1

    def set_frame(self, frame):
        if not self._closed:
            if self._frame_queue.qsize() <= 2:
                self._frame_queue.put(frame)

    @staticmethod
    def _process_fn(args_queue, frame_queue, result_queue):
        args = dnnlib.EasyDict()
        stamp = 0
        curr_args = dnnlib.EasyDict()
        pipeline_obj = None
        while True:
            if args_queue.qsize() > 0:
                args, stamp = args_queue.get()
                # Delete args.prompt
                if 'prompt' in args:
                    del args['prompt']
                if not compare_args(args, curr_args):
                    try:
                        pipeline_obj = pipeline.Pipeline(**args)
                    except Exception as e:
                        result_queue.put([dnnlib.EasyDict({'error': str(e)}), stamp])
                    curr_args = args

            if frame_queue.qsize() > 0 and pipeline_obj is not None:
                frame = frame_queue.get()
                frame = frame[:, :, :3]
                # print(frame.shape, frame.dtype)
                input_image = Image.fromarray(frame)
                args, stamp = args_queue.get()
                result = pipeline_obj.predict(input_image, args['prompt'])

                if 'error' in result:
                    result.error = pipeline.CapturedException(result.error)
                result_queue.put([result, stamp])
                del result
            # gc.collect() # Putting a garbage collect here stabilizes the memory usage, but slows down the rendering
            # Torch seems to store values in the background even with nograd that slow down StyleGAN2 over time
            # This is a workaround to keep the memory usage stable, but conflicts with imgui causing drops in GUI performance
            torch.cuda.empty_cache()
