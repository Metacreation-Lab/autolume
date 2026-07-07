import gc
import multiprocessing
import torch

import dnnlib
from utils import device_utils
from utils.app_logging import LoggedProcess
from widgets import renderer

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
            if not(torch.equal(a1, a2)):
                return False
        else:
            if not (a1 == a2):
                return False
    return True

class AsyncRenderer:
    def __init__(self):
        self._closed        = False
        self._cur_args      = None
        self._cur_result    = None
        self._cur_stamp     = 0
        self._args_queue = multiprocessing.Queue()
        self._result_queue = multiprocessing.Queue()
        self._process = LoggedProcess(target=self._process_fn, args=(self._args_queue, self._result_queue),
                                      daemon=True, name='renderer')
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
            if self._args_queue.empty():
                if not compare_args(args, self._cur_args):
                    self._args_queue.put([args, self._cur_stamp])
                self._cur_args = args

    def get_result(self):
        if not self._closed:
            if self._result_queue is not None:
                if not self._result_queue.empty():
                    result, stamp = self._result_queue.get()
                    while not self._result_queue.empty():
                        result, stamp = self._result_queue.get()
                    self._cur_result = result
            return self._cur_result

    def clear_result(self):
        assert not self._closed
        self._cur_args = None
        self._cur_result = None
        self._cur_stamp += 1

    @staticmethod
    def _process_fn(args_queue, result_queue):
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_grad_enabled(False)
        renderer_obj = renderer.Renderer()
        renders_since_flush = 0
        with torch.inference_mode():
            while True:
                # Block instead of polling: a hot loop pins a CPU core, which
                # steals GPU boost headroom on power-limited machines.
                args, stamp = args_queue.get()
                while not args_queue.empty():
                    args, stamp = args_queue.get()
                with torch.no_grad():
                    result = renderer_obj.render(**args)
                if 'error' in result:
                    result.error = renderer.CapturedException(result.error)
                result_queue.put([result, stamp])
                del result
                renders_since_flush += 1
                # gc.collect() # Putting a garbage collect here stabilizes the memory usage, but slows down the rendering
                               # Torch seems to store values in the background even with nograd that slow down StyleGAN2 over time
                               # This is a workaround to keep the memory usage stable, but conflicts with imgui causing drops in GUI performance
                # empty_cache() synchronizes the GPU on both CUDA and MPS
                # (~5 ms/frame on CUDA); flush periodically to keep memory
                # bounded instead.
                if renders_since_flush >= 120:
                    device_utils.empty_cache()
                    renders_since_flush = 0
