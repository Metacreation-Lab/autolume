"""Naming the config a bare .safetensors needs.

diffusers fetches a reference repo instead of reading the file, and the SD 2.x
ones were deleted upstream, so sd-turbo stopped loading.
"""
import json
import struct

from diffusion import single_file


def write_checkpoint(path, context_dim, key="lora_unet_x_attn2_to_k.lora_down.weight"):
    header = {key: {"dtype": "F16", "shape": [8, context_dim], "data_offsets": [0, 0]}}
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob)
    return str(path)


def test_sd15_needs_no_help(tmp_path):
    """Its reference repo is still up, so diffusers finds it unaided."""
    path = write_checkpoint(tmp_path / "a.safetensors", 768)
    assert single_file.context_dim(path) == 768
    assert single_file.config_for(path) is None


def test_sd2x_is_pointed_at_a_config_that_still_exists(tmp_path):
    path = write_checkpoint(tmp_path / "a.safetensors", 1024)
    assert single_file.context_dim(path) == 1024
    assert single_file.config_for(path) == "stabilityai/sd-turbo"


def test_a_model_id_is_left_alone(tmp_path):
    assert single_file.config_for("stabilityai/sd-turbo") is None


def test_an_unreadable_file_is_not_fatal(tmp_path):
    junk = tmp_path / "junk.safetensors"
    junk.write_bytes(b"not a safetensors file at all")
    assert single_file.context_dim(str(junk)) is None
    assert single_file.config_for(str(junk)) is None
    assert single_file.config_for(str(tmp_path / "missing.safetensors")) is None


def patched(cls):
    """from_single_file is inherited, so ours is visible as a class attribute."""
    return "from_single_file" in cls.__dict__


def test_the_override_is_scoped_and_always_restored(tmp_path):
    from diffusers import StableDiffusionPipeline
    path = write_checkpoint(tmp_path / "a.safetensors", 1024)
    assert not patched(StableDiffusionPipeline)
    with single_file.config_override(path):
        assert patched(StableDiffusionPipeline)
        assert StableDiffusionPipeline.from_single_file.keywords["config"] == "stabilityai/sd-turbo"
    # inheritance restored, not overwritten with a copy bound to this class
    assert not patched(StableDiffusionPipeline)
    try:
        with single_file.config_override(path):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert not patched(StableDiffusionPipeline)


def test_nothing_is_patched_when_no_config_is_needed(tmp_path):
    from diffusers import StableDiffusionPipeline
    with single_file.config_override(write_checkpoint(tmp_path / "a.safetensors", 768)):
        assert not patched(StableDiffusionPipeline)


def test_sdxl_is_left_alone(tmp_path):
    """Only the SD pipeline is touched; nothing else changes behaviour."""
    from diffusers import StableDiffusionXLPipeline
    with single_file.config_override(write_checkpoint(tmp_path / "a.safetensors", 1024)):
        assert "from_single_file" not in StableDiffusionXLPipeline.__dict__
