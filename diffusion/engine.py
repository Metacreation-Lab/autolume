"""Headless StreamDiffusion img2img stage. UI-free so it is unit-testable
and importable on platforms where the streamdiffusion extra is absent."""
import importlib.util

import torch
import torch.nn.functional as F

NUM_INFERENCE_STEPS = 50


def is_available():
    return importlib.util.find_spec("streamdiffusion") is not None


def default_params():
    return dict(model="stabilityai/sd-turbo", prompt="", strength=0.5, seed=0,
                lora_path="", lora_scale=1.0, acceleration="none", resolution=512)


def t_indices_for_strength(strength, num_steps=50, steps=1):
    strength = min(max(float(strength), 0.0), 1.0)
    start = round((1.0 - strength) * (num_steps - 1))
    if steps == 1:
        return [start]
    stride = max(1, (num_steps - 1 - start) // steps)
    idx = [min(start + i * stride, num_steps - 1) for i in range(steps)]
    return sorted(set(idx))


def to_diffusion_input(out, resolution):
    x = out[:, :3] if out.shape[1] >= 3 else out[:, :1].repeat(1, 3, 1, 1)
    x = x.to(torch.float32).clamp(-1, 1).add(1).div(2)
    if x.shape[-2:] != (resolution, resolution):
        x = F.interpolate(x, size=(resolution, resolution), mode="bilinear",
                          antialias=True, align_corners=False)
    return x


def from_diffusion_output(img):
    return img.to(torch.float32).clamp(0, 1) * 2 - 1


def build_key(params):
    return (params["model"], params["acceleration"], params["resolution"],
            params["lora_path"], params["lora_scale"])


def _make_wrapper(params, device):
    from streamdiffusion import StreamDiffusionWrapper

    lora_dict = {params["lora_path"]: params["lora_scale"]} if params["lora_path"] else None
    return StreamDiffusionWrapper(
        model_id_or_path=params["model"],
        t_index_list=t_indices_for_strength(params["strength"], NUM_INFERENCE_STEPS),
        mode="img2img",
        acceleration=params["acceleration"],
        output_type="pt",
        width=params["resolution"],
        height=params["resolution"],
        frame_buffer_size=1,
        lora_dict=lora_dict,
        device=str(device),
        seed=int(params["seed"]),
        # the fork's default cfg_type "self" diverges to NaN after ~45 frames and never recovers
        cfg_type="none",
    )


def _error_status(exc):
    lines = str(exc).splitlines()
    return f"Error: {lines[0] if lines else type(exc).__name__}"


class DiffusionEngine:
    def __init__(self):
        self._wrapper = None
        self._key = None
        self._applied = None
        self.status = ""

    def process(self, out, params, device):
        key = build_key(params)
        if key != self._key:
            self._wrapper = None
            self._applied = None
            self._key = key
            self.status = ""
            try:
                self._wrapper = _make_wrapper(params, device)
            except Exception as e:
                self.status = _error_status(e)
        if self._wrapper is None:
            return out
        try:
            self._apply(params)
            x = to_diffusion_input(out, params["resolution"]).to(device)
            img = self._wrapper(image=x)
            self.status = ""
            return from_diffusion_output(img).to(out.dtype)
        except Exception as e:
            self.status = _error_status(e)
            return out

    def _apply(self, params):
        applied = (params["prompt"], params["strength"], params["seed"])
        if applied == self._applied:
            return
        if self._applied is None:
            self._wrapper.prepare(prompt=params["prompt"],
                                  num_inference_steps=NUM_INFERENCE_STEPS)
            self._update_stream_params(params)
        else:
            if applied[0] != self._applied[0]:
                self._wrapper.update_prompt(params["prompt"])
            if applied[1:] != self._applied[1:]:
                self._update_stream_params(params)
        self._applied = applied

    def _update_stream_params(self, params):
        self._wrapper.update_stream_params(
            t_index_list=t_indices_for_strength(params["strength"], NUM_INFERENCE_STEPS),
            seed=int(params["seed"]))
