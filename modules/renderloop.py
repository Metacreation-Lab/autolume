import gc
import multiprocessing
import torch

import dnnlib
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
        self._process = multiprocessing.Process(target=self._process_fn, args=(self._args_queue, self._result_queue),
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

    @staticmethod
    def _process_fn(args_queue, result_queue):
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_grad_enabled(False)
        renderer_obj = renderer.Renderer()
        args = None
        stamp = 0
        new_arg = False
        with torch.inference_mode():
            while True:
                if args_queue.qsize() > 0:
                    args, stamp = args_queue.get()
                    new_arg = True
                if new_arg:
                    with torch.no_grad():
                        result = renderer_obj.render(**args)
                    if 'error' in result:
                        result.error = renderer.CapturedException(result.error)
                    result_queue.put([result, stamp])
                    del result
                    new_arg = False
                # gc.collect() # Putting a garbage collect here stabilizes the memory usage, but slows down the rendering
                               # Torch seems to store values in the background even with nograd that slow down StyleGAN2 over time
                               # This is a workaround to keep the memory usage stable, but conflicts with imgui causing drops in GUI performance
                torch.cuda.empty_cache()
