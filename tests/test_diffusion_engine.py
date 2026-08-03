import pytest
import torch

from diffusion import engine


def test_availability_probe_is_bool():
    assert isinstance(engine.is_available(), bool)


def test_strength_maps_monotonically_and_in_bounds():
    prev = None
    for s in [0.0, 0.25, 0.5, 0.75, 1.0]:
        idx = engine.t_indices_for_strength(s)
        assert len(idx) == 1
        assert 0 <= idx[0] <= 49
        if prev is not None:
            assert idx[0] <= prev  # more strength = earlier index = more transformation
        prev = idx[0]


def test_strength_two_step_is_sorted_unique():
    idx = engine.t_indices_for_strength(0.6, steps=2)
    assert idx == sorted(set(idx)) and len(idx) == 2


def test_to_diffusion_input_shapes_and_range():
    out = torch.linspace(-2, 2, 1 * 3 * 64 * 64).reshape(1, 3, 64, 64)
    x = engine.to_diffusion_input(out, resolution=32)
    assert x.shape == (1, 3, 32, 32)
    assert x.min() >= 0.0 and x.max() <= 1.0


def test_to_diffusion_input_single_channel_repeats():
    out = torch.zeros(1, 1, 64, 64)
    assert engine.to_diffusion_input(out, 32).shape == (1, 3, 32, 32)


def test_to_diffusion_input_many_channels_truncates():
    out = torch.zeros(1, 7, 64, 64)
    assert engine.to_diffusion_input(out, 32).shape == (1, 3, 32, 32)


def test_range_round_trip():
    img = torch.rand(1, 3, 8, 8)
    back = engine.from_diffusion_output(img)
    assert back.min() >= -1.0001 and back.max() <= 1.0001


def test_build_key_ignores_performance_params():
    a = engine.default_params()
    b = dict(a, prompt="x", strength=0.9, seed=123)
    assert engine.build_key(a) == engine.build_key(b)


def test_build_key_changes_on_model_and_lora():
    a = engine.default_params()
    assert engine.build_key(a) != engine.build_key(dict(a, model="other/model"))
    assert engine.build_key(a) != engine.build_key(dict(a, lora_path="x.safetensors"))
