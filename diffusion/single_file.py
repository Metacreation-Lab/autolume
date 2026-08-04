"""Telling diffusers what a bare .safetensors checkpoint is.

A single file carries weights but no architecture config. ComfyUI infers one
from the tensor shapes; diffusers instead downloads a reference repo from
HuggingFace, and Stability deleted the SD 2.x ones, so sd-turbo stopped loading
with a 401 for a repo the user never asked for.

The architecture is right there in the file's header, so we read it and name a
config that still exists. SD 1.5 needs none of this: its reference repo is up,
and diffusers finds it on its own.
"""
import contextlib
import json
import logging
import re
import struct
from functools import partial

logger = logging.getLogger(__name__)

# Cross attention takes the text embedding, so the width of attn2's key/value
# projection is the text encoder width, which identifies the architecture.
ATTN2_KV = re.compile(r"attn2[._]to_[kv]")

# Only architectures whose own reference repo is missing need an answer here.
CONFIG_FOR_CONTEXT_DIM = {
    1024: "stabilityai/sd-turbo",  # SD 2.x: stabilityai/stable-diffusion-2-1 now 401s
}


def context_dim(path):
    """Text encoder width from a safetensors header, or None if unreadable.

    Reads only the header, which is a few hundred KB even on a 5 GB file.
    """
    try:
        with open(path, "rb") as f:
            length = struct.unpack("<Q", f.read(8))[0]
            if length > 100_000_000:
                return None
            header = json.loads(f.read(length))
    except (OSError, ValueError, struct.error):
        return None
    widths = [tensor["shape"][1] for name, tensor in header.items()
              if ATTN2_KV.search(name) and len(tensor.get("shape") or []) == 2]
    return max(widths) if widths else None


def config_for(path):
    """Config repo a single file needs, or None to let diffusers decide."""
    if not str(path).lower().endswith(".safetensors"):
        return None
    return CONFIG_FOR_CONTEXT_DIM.get(context_dim(path))


@contextlib.contextmanager
def config_override(path):
    """Supply the config while a checkpoint loads.

    The fork calls ``from_single_file(path)`` with no arguments, so there is
    nowhere else to put it.
    """
    config = config_for(path)
    if config is None:
        yield
        return
    from diffusers import StableDiffusionPipeline

    logger.info("Loading %s with the %s config", path, config)
    # from_single_file is inherited from a mixin, so restoring means removing
    # what we set rather than assigning the method back: assigning would leave
    # a copy bound to this class on a class that never had one.
    owned = StableDiffusionPipeline.__dict__.get("from_single_file")
    StableDiffusionPipeline.from_single_file = partial(
        StableDiffusionPipeline.from_single_file, config=config)
    try:
        yield
    finally:
        if owned is None:
            del StableDiffusionPipeline.from_single_file
        else:
            StableDiffusionPipeline.from_single_file = owned
