import gc
import multiprocessing
import torch
from torchvision.io import read_video
import dnnlib
from widgets import pipeline
from tqdm import tqdm

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
        pipeline_obj = pipeline.Pipeline()
        args = None
        stamp = 0
        new_arg = False
        while True:
            if args_queue.qsize() > 0:
                args, stamp = args_queue.get()
                new_arg = True
            if new_arg:
                video_info = read_video(args.input[0])
                video = video_info[0] / 255
                fps = video_info[2]["video_fps"]
                height = int(video.shape[1] * args.scale)
                width = int(video.shape[2] * args.scale)

                for _ in range(pipeline_obj.stream.batch_size):
                    pipeline_obj.stream(image=video[0].permute(2, 0, 1))

                for i in tqdm(range(video.shape[0])):
                    input_image = video[i].permute(2, 0, 1)
                    result = pipeline_obj.predict(input_image, **args)

                    if 'error' in result:
                        result.error = pipeline.CapturedException(result.error)
                    result_queue.put([result, stamp])
                    del result
                new_arg = False
            # gc.collect() # Putting a garbage collect here stabilizes the memory usage, but slows down the rendering
            # Torch seems to store values in the background even with nograd that slow down StyleGAN2 over time
            # This is a workaround to keep the memory usage stable, but conflicts with imgui causing drops in GUI performance
            torch.cuda.empty_cache()
