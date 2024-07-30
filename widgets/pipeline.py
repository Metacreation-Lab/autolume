import sys
import traceback
import dnnlib
import torch
from torchvision.io import read_video
from utils.wrapper import StreamDiffusionWrapper
import numpy as np

default_prompt = "Portrait of The Joker halloween costume, face painting, with , glare pose, detailed, intricate, full of colour"
default_negative_prompt = "black and white, blurry, low resolution, pixelated,  pixel art, low quality, low fidelity"


class Pipeline:
    def __init__(self, **args):
        self.stream = StreamDiffusionWrapper(**args)
        self.last_prompt = default_prompt
        self.res = dnnlib.EasyDict()
        self.stream.prepare(
            prompt=default_prompt,
            negative_prompt=default_negative_prompt,
            num_inference_steps=50,
            guidance_scale=1.2,
        )

    def predict(self, image, prompt):
        # if prompt != self.last_prompt:
        #     self.last_prompt = prompt
        #     self.stream.prepare(
        #         prompt=self.last_prompt,
        #         negative_prompt=default_negative_prompt,
        #         num_inference_steps=50,
        #         guidance_scale=1.2,
        #     )
        try:
            image_tensor = self.stream.preprocess_image(image)
            output_image = self.stream(image=image_tensor, prompt=prompt)
            self.res.image = output_image
        except:
            self.res.error = CapturedException()
        if 'image' in self.res:
            # print(res.image.shape, res.image.dtype)
            # # Move the image tensor to CPU and convert it to a NumPy ndarray
            # res.image = res.image.cpu().permute(1, 2, 0).numpy()
            pass
        if 'error' in self.res:
            self.res.error = str(self.res.error)
        return self.res

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
