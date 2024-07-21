import sys
import traceback
import dnnlib
import torch
from torchvision.io import read_video
from utils.wrapper import StreamDiffusionWrapper
import numpy as np

default_prompt = "girl with brown dog ears, thick frame glasses"


class Pipeline:
    def __init__(self, **args):
        self.stream = StreamDiffusionWrapper(
            model_id_or_path=args.get("model_id", "stabilityai/sd-turbo"),
            frame_buffer_size=1,
            warmup=10,
            acceleration="xformers",
            mode="img2img",
            t_index_list=[35, 45],
            # output_type="pil",
            use_denoising_batch=True,
            cfg_type="none",
            use_lcm_lora=False,
        )
        self.last_prompt = default_prompt
        self.stream.prepare(
            prompt=default_prompt,
            num_inference_steps=50,
        )

    def predict(self, image, **args):
        res = dnnlib.EasyDict()
        try:
            image_tensor = self.stream.preprocess_image(image)
            output_image = self.stream(image=image_tensor, prompt=args['prompt'])
            res.image = output_image
        except:
            res.error = CapturedException()
        if 'image' in res:
            # print(res.image.shape, res.image.dtype)
            # # Move the image tensor to CPU and convert it to a NumPy ndarray
            # res.image = res.image.cpu().permute(1, 2, 0).numpy()
            pass
        if 'error' in res:
            res.error = str(res.error)
        return res

    def to_cpu(self, tensor):
        return tensor.detach().cpu()

    def _render_impl(self, res,
                     input="",
                     model_id="KBlueLeaf/kohaku-v2.1",
                     lora_dict=None,
                     prompt="1girl with brown dog ears, thick frame glasses",
                     scale=1.0,
                     acceleration="xformers",
                     use_denoising_batch=True,
                     enable_similar_image_filter=True,
                     seed=2
                     ):
        print(input)
        video_info = read_video(input)
        video = video_info[0] / 255
        height = int(video.shape[1] * scale)
        width = int(video.shape[2] * scale)

        for _ in range(self.stream.batch_size):
            self.stream(image=video[0].permute(2, 0, 1))

        res.progress = 0.0
        for i in range(video.shape[0]):
            img = self.stream(video[i].permute(2, 0, 1))
            res.image = img
            res.progress = (i + 1) / video.shape[0]
            del img


class CapturedException(Exception):
    def __init__(self, msg=None):
        if msg is None:
            _type, value, _traceback = sys.exc_info()
            assert value is not None
            if isinstance(value, CapturedException):
                msg = str(value)
            else:
                msg = traceback.format_exc()
        assert isinstance(msg, str)
        super().__init__(msg)
