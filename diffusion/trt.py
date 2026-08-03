"""TensorRT engine management for the diffusion stage. Builds run in a
separate LoggedProcess because a first build takes 20 to 30 minutes."""
import hashlib
import importlib.util
import os

from utils import user_data

FORK_SHA = "4c90d9e437aa28cca7cae1acfab1e52157261939"

# Engines are compiled per model, resolution and LoRA. The fork's own engine
# directory names cover neither the resolution nor its own version, so they hang
# under this key instead.
REQUIRED_ENGINES = ("unet.engine", "vae_encoder.engine", "vae_decoder.engine")

# tensorrt itself is installed by streamdiffusion.tools.install-tensorrt, which
# shells out to uv pip and mutates the venv behind uv's back. Autolume never runs
# it, so a missing tensorrt is reported instead of worked around.
NOT_INSTALLED = "TensorRT is not installed. Run the install tool first."


def engine_dir_key(params):
    raw = "|".join([params["model"], str(params["resolution"]),
                    params["lora_path"], f'{params["lora_scale"]:.3f}', FORK_SHA])
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def engine_dir(params):
    return str(user_data.data_path("trt-engines", engine_dir_key(params)))


def engines_ready(params):
    root = engine_dir(params)
    if not os.path.isdir(root):
        return False
    found = set()
    for _dirpath, _dirnames, filenames in os.walk(root):
        found.update(filenames)
    return all(name in found for name in REQUIRED_ENGINES)


def run_build(cmd_queue, reply_queue):
    """LoggedProcess entry point.

    cmd_queue: {'cmd': 'build', 'params': dict} | {'cmd': 'shutdown'}
    reply_queue: {'progress': str} ... {'done': True} | {'error': str}
    """
    try:
        while True:
            msg = cmd_queue.get()
            if msg.get("cmd") == "shutdown":
                return
            if msg.get("cmd") != "build":
                continue
            if importlib.util.find_spec("tensorrt") is None:
                reply_queue.put({"error": NOT_INSTALLED})
                continue
            params = dict(msg["params"], acceleration="tensorrt")
            reply_queue.put({"progress": "Loading the model"})

            from diffusion.engine import wrapper_kwargs
            from utils import device_utils

            os.makedirs(engine_dir(params), exist_ok=True)
            kwargs = wrapper_kwargs(params, device_utils.get_device())
            kwargs.update(build_engines_if_missing=True, compile_engines_only=True)

            from streamdiffusion import StreamDiffusionWrapper

            reply_queue.put({"progress": "Exporting ONNX and building engines"})
            StreamDiffusionWrapper(**kwargs)
            reply_queue.put({"progress": "Finishing"})
            reply_queue.put({"done": True})
    except Exception:
        import traceback
        reply_queue.put({"error": traceback.format_exc()})
