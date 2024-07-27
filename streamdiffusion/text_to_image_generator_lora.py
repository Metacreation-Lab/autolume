import torch
from diffusers import StableDiffusionPipeline


class ImageGenerator:
    def __init__(self):
        self.pipe = None

    def predict(self,
                model_path="",
                pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4",
                prompt="Joker from Batman", save_path="joker.png"):
        if self.pipe is None:
            self.pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path,
                                                                torch_dtype=torch.float16)
            if model_path != "":
                self.pipe.unet.load_attn_procs(model_path)
            self.pipe.to("cuda")

        image = self.pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
        image.save(save_path)
