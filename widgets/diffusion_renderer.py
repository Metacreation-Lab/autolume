import sys
import traceback
import dnnlib
import torch
from torchvision.io import read_video
from utils.wrapper import StreamDiffusionWrapper


class Renderer:
    def __init__(self):
        self.stream = None

    def render(self, **args):
        res = dnnlib.EasyDict()
        try:
            self._render_impl(res, **args)
        except:
            res.error = CapturedException()
        if 'image' in res:
            res.image = self.to_cpu(res.image).numpy()
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
        video_info = read_video(input)
        video = video_info[0] / 255
        height = int(video.shape[1] * scale)
        width = int(video.shape[2] * scale)

        if self.stream is None:
            self.stream = StreamDiffusionWrapper(
                model_id_or_path=model_id,
                lora_dict=lora_dict,
                t_index_list=[35, 45],
                frame_buffer_size=1,
                width=width,
                height=height,
                warmup=10,
                acceleration=acceleration,
                do_add_noise=False,
                mode="img2img",
                output_type="pt",
                enable_similar_image_filter=enable_similar_image_filter,
                similar_image_filter_threshold=0.98,
                use_denoising_batch=use_denoising_batch,
                seed=seed,
            )

        self.stream.prepare(
            prompt=prompt,
            num_inference_steps=50,
        )

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
