"""TensorRT engine management for the diffusion stage. Builds run in a
separate LoggedProcess because a first build takes 10 to 20 minutes."""
import hashlib
import importlib.util
import json
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


def _dir_has_engines(root):
    if not os.path.isdir(root):
        return False
    found = set()
    for _dirpath, _dirnames, filenames in os.walk(root):
        found.update(filenames)
    return all(name in found for name in REQUIRED_ENGINES)


def engines_ready(params):
    return _dir_has_engines(engine_dir(params))


# The engine dir name is an opaque hash, so a manifest of the build params is
# written next to the engines to make built sets listable and re-selectable.
MANIFEST_KEYS = ("model", "resolution", "lora_path", "lora_scale")


def write_manifest(params):
    manifest = {key: params[key] for key in MANIFEST_KEYS}
    manifest["fork_sha"] = FORK_SHA
    with open(os.path.join(engine_dir(params), "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def list_built_engines():
    """Build params of every complete engine set, sorted by checkpoint name.

    Sets built by an older fork pin are skipped: their key no longer matches
    what the current code would compute, so they cannot be loaded.
    """
    root = str(user_data.data_path("trt-engines"))
    if not os.path.isdir(root):
        return []
    entries = []
    for name in sorted(os.listdir(root)):
        subdir = os.path.join(root, name)
        manifest_path = os.path.join(subdir, "manifest.json")
        if not os.path.isfile(manifest_path) or not _dir_has_engines(subdir):
            continue
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            entry = {key: manifest[key] for key in MANIFEST_KEYS}
        except (OSError, ValueError, KeyError):
            continue
        if manifest.get("fork_sha") != FORK_SHA:
            continue
        entries.append(entry)
    return sorted(entries, key=lambda e: str(e["model"]).lower())


def build_progress(root, elapsed):
    """Human-readable build stage, derived from what is on disk.

    TensorRT exposes no percentage, so progress is reported from the artifacts
    each stage leaves behind: an .onnx export, then a compiled .engine.
    """
    done, exporting = set(), set()
    for _dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(".engine"):
                done.add(name)
            elif name.endswith(".onnx"):
                exporting.add(os.path.splitext(name)[0])
    minutes = int(elapsed // 60)
    suffix = f" - {minutes} min elapsed" if minutes else ""
    if len(done) >= len(REQUIRED_ENGINES):
        return f"Finishing{suffix}"
    stage = sorted(name for name in REQUIRED_ENGINES if name not in done)[0]
    stage = stage[: -len(".engine")]
    verb = "Compiling" if stage in exporting else "Exporting"
    return f"{verb} {stage} ({len(done) + 1} of {len(REQUIRED_ENGINES)}){suffix}"


def _watch_progress(root, reply_queue, stop):
    import time
    started = time.time()
    last = None
    while not stop.is_set():
        message = build_progress(root, time.time() - started)
        if message != last:
            reply_queue.put({"progress": message})
            last = message
        stop.wait(2.0)


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

            # kwargs first: it validates the params, and a rejected build should
            # not leave an empty engine dir behind
            kwargs = wrapper_kwargs(params, device_utils.get_device())
            kwargs.update(build_engines_if_missing=True, compile_engines_only=True)
            os.makedirs(engine_dir(params), exist_ok=True)

            from streamdiffusion import StreamDiffusionWrapper

            import threading
            stop = threading.Event()
            watcher = threading.Thread(target=_watch_progress, daemon=True,
                                       args=(engine_dir(params), reply_queue, stop))
            watcher.start()
            try:
                StreamDiffusionWrapper(**kwargs)
            finally:
                stop.set()
            write_manifest(params)
            reply_queue.put({"done": True})
    except Exception:
        import traceback
        reply_queue.put({"error": traceback.format_exc()})
