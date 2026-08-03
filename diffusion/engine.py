"""Headless StreamDiffusion img2img stage. UI-free so it is unit-testable
and importable on platforms where the streamdiffusion extra is absent."""
import importlib.util
import os
import threading
import time

import torch
import torch.nn.functional as F

NUM_INFERENCE_STEPS = 50

TRT_NOT_BUILT = "TensorRT engines not built. Use Build engines."

LOADING_PREFIX = "Loading pipeline"


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


def wrapper_kwargs(params, device):
    lora_path = params["lora_path"]
    kwargs = dict(
        model_id_or_path=params["model"],
        t_index_list=t_indices_for_strength(params["strength"], NUM_INFERENCE_STEPS),
        mode="img2img",
        acceleration=params["acceleration"],
        output_type="pt",
        width=params["resolution"],
        height=params["resolution"],
        frame_buffer_size=1,
        lora_dict={lora_path: params["lora_scale"]} if lora_path else None,
        device=str(device),
        seed=int(params["seed"]),
        # the fork's default cfg_type "self" diverges to NaN after ~45 frames and never recovers
        cfg_type="none",
        # non-distilled checkpoints produce half-denoised mush at the few steps this
        # stage runs; the fork fuses lcm-lora-sdv1-5 for them and skips sd-turbo itself
        use_lcm_lora=True,
    )
    if params["acceleration"] == "tensorrt":
        from diffusion import trt
        kwargs["engine_dir"] = trt.engine_dir(params)
        # a 20 to 30 minute build must never happen inside the render worker
        kwargs["build_engines_if_missing"] = False
    return kwargs


def _make_wrapper(params, device):
    lora_path = params["lora_path"]
    # the fork logs and swallows a failed LoRA load, so a bad path would otherwise
    # build a silently un-LoRA'd pipeline
    if lora_path and not os.path.isfile(lora_path):
        raise FileNotFoundError(f"LoRA file not found: {lora_path}")

    from streamdiffusion import StreamDiffusionWrapper

    return StreamDiffusionWrapper(**wrapper_kwargs(params, device))


def _error_status(exc):
    lines = str(exc).splitlines()
    return f"Error: {lines[0] if lines else type(exc).__name__}"


class DiffusionEngine:
    """Pipelines load in a background thread so the render worker keeps
    producing (undiffused) frames instead of freezing for the 3 to 40 s a
    build takes; the finished pipeline is swapped in between frames."""

    def __init__(self):
        self._wrapper = None
        self._key = None
        self._applied = None
        self._loader = None
        self.status = ""

    @property
    def loading(self):
        """True while a pipeline is being built on the loader thread.

        The render loop renders on demand, so it has to keep producing frames
        while this is set or the finished pipeline would never be installed.
        """
        return self._loader is not None

    def _start_load(self, params, device, key):
        box = {}
        snapshot = dict(params)

        def work():
            try:
                box["wrapper"] = _make_wrapper(snapshot, device)
            except Exception as e:
                box["error"] = e

        thread = threading.Thread(target=work, daemon=True, name="diffusion-load")
        self._loader = dict(key=key, thread=thread, box=box, started=time.time())
        thread.start()

    def _sweep_loader(self, key):
        """Install a finished load for the current key; discard a stale one."""
        if self._loader is None or self._loader["thread"].is_alive():
            return
        loader = self._loader
        self._loader = None
        if loader["key"] != key:
            loader["box"].pop("wrapper", None)
            return
        self._key = key  # errors latch: no retry until the key changes
        if "error" in loader["box"]:
            self.status = _error_status(loader["box"]["error"])
        else:
            self._wrapper = loader["box"]["wrapper"]
            self._applied = None
            self.status = ""

    def process(self, out, params, device):
        key = build_key(params)
        self._sweep_loader(key)
        if params["acceleration"] == "tensorrt":
            from diffusion import trt
            # falling back to the eager pipeline here would silently cost the user
            # the speed they asked for, so the stage passes frames through instead
            if not trt.engines_ready(params):
                self._wrapper = None
                self._applied = None
                self._key = None
                self.status = TRT_NOT_BUILT
                return out
        if key != self._key and self._loader is None:
            # release the old pipeline first so the load never doubles VRAM
            self._wrapper = None
            self._applied = None
            self.status = f"{LOADING_PREFIX} (0 s)"
            self._start_load(params, device, key)
            return out
        if self._loader is not None:
            self.status = f"{LOADING_PREFIX} ({int(time.time() - self._loader['started'])} s)"
            return out
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
