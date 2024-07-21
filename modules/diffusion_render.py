import gc
import multiprocessing
import torch
from torchvision.io import read_video
import dnnlib
from widgets import pipeline
from tqdm import tqdm
from PIL import Image
import numpy as np

stabilityai_sd_turbo_args = dnnlib.EasyDict({
    "model_id_or_path": "stabilityai/sd-turbo",
    "frame_buffer_size": 1,
    "warmup": 10,
    "acceleration": "xformers",
    "mode": "img2img",
    "t_index_list": [35, 45],
    "output_type": "np",
    "use_denoising_batch": True,
    "cfg_type": "none",
    "use_lcm_lora": False,
})

KBlueLeaf_kohaku_v2_1_args = dnnlib.EasyDict({
    "model_id_or_path": "KBlueLeaf/kohaku-v2.1",
    "lora_dict": None,
    "t_index_list": [35, 45],
    "frame_buffer_size": 1,
    "warmup": 10,
    "acceleration": "xformers",
    "do_add_noise": False,
    "mode": "img2img",
    "output_type": "np",
    "enable_similar_image_filter": True,
    "similar_image_filter_threshold": 0.98,
    "use_denoising_batch": True,
    "seed": 2,
})

def compare_args(args, cur_args):
    if args is None or cur_args is None:
        return False
    if args.keys() != cur_args.keys():
        return False
    for key in args.keys():
        a1 = args.get(key, "a")
        a2 = cur_args.get(key, "b")
        if not isinstance(a1, type(a2)):
            return False
        if isinstance(a1, dict):
            if not compare_args(a1, a2):
                return False
        elif isinstance(a1, torch.Tensor):
            if not (torch.equal(a1, a2)):
                return False
        else:
            if not (a1 == a2):
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
                if not compare_args(args, self._cur_args):
                    self._args_queue.put([args, self._cur_stamp])
                self._cur_args = args

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
        args = None
        stamp = 0
        curr_model_id = None
        pipeline_obj = None
        while True:
            if args_queue.qsize() > 0:
                args, stamp = args_queue.get()
                if curr_model_id != args['model_id']:
                    curr_model_id = args['model_id']
                    if curr_model_id == "stabilityai/sd-turbo":
                        pipeline_obj = pipeline.Pipeline(**stabilityai_sd_turbo_args)
                    elif curr_model_id == "KBlueLeaf/kohaku-v2.1":
                        pipeline_obj = pipeline.Pipeline(**KBlueLeaf_kohaku_v2_1_args)

            if frame_queue.qsize() > 0 and pipeline_obj is not None:
                frame = frame_queue.get()
                frame = frame[:, :, :3]
                # print(frame.shape, frame.dtype)
                input_image = Image.fromarray(frame)
                result = pipeline_obj.predict(input_image, **args)

                if 'error' in result:
                    result.error = pipeline.CapturedException(result.error)
                result_queue.put([result, stamp])
                del result
            # gc.collect() # Putting a garbage collect here stabilizes the memory usage, but slows down the rendering
            # Torch seems to store values in the background even with nograd that slow down StyleGAN2 over time
            # This is a workaround to keep the memory usage stable, but conflicts with imgui causing drops in GUI performance
            torch.cuda.empty_cache()
