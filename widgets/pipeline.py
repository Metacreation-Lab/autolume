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
