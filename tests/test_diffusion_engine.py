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


class FakeWrapper:
    def __init__(self):
        self.prompts, self.prepares, self.stream_params, self.calls = [], [], [], []
        self.frames = 0
        self.fail_on_frame = False

    def prepare(self, prompt, negative_prompt="", num_inference_steps=50, **kwargs):
        self.calls.append("prepare")
        self.prepares.append((prompt, num_inference_steps))

    def update_prompt(self, prompt, **kwargs):
        self.calls.append("update_prompt")
        self.prompts.append(prompt)

    def update_stream_params(self, **kwargs):
        self.calls.append("update_stream_params")
        self.stream_params.append(kwargs)

    def __call__(self, image=None, prompt=None):
        self.calls.append("frame")
        self.frames += 1
        if self.fail_on_frame:
            raise RuntimeError("boom on frame")
        return torch.rand(1, 3, image.shape[-2], image.shape[-1], dtype=torch.float16)


def make_engine(monkeypatch, factory=None):
    calls = []

    def fake_make(params, device):
        calls.append(engine.build_key(params))
        if factory:
            return factory(params, device)
        return FakeWrapper()

    monkeypatch.setattr(engine, "_make_wrapper", fake_make)
    return engine.DiffusionEngine(), calls


def holding_engine(monkeypatch):
    holder = {}

    def factory(params, device):
        holder["w"] = FakeWrapper()
        return holder["w"]

    eng, calls = make_engine(monkeypatch, factory)
    return eng, holder, calls


def test_process_returns_diffused_frame_in_stylegan_range(monkeypatch):
    eng, _ = make_engine(monkeypatch)
    out = torch.zeros(1, 3, 64, 64)
    res = eng.process(out, engine.default_params(), torch.device("cpu"))
    assert res.shape == (1, 3, 512, 512)
    assert res.min() >= -1.0001 and res.max() <= 1.0001
    assert eng.status == ""


def test_no_rebuild_on_prompt_or_strength_change(monkeypatch):
    eng, calls = make_engine(monkeypatch)
    p = engine.default_params()
    eng.process(torch.zeros(1, 3, 64, 64), p, torch.device("cpu"))
    eng.process(torch.zeros(1, 3, 64, 64), dict(p, prompt="new", strength=0.9), torch.device("cpu"))
    assert len(calls) == 1


def test_no_rebuild_on_seed_change(monkeypatch):
    eng, calls = make_engine(monkeypatch)
    p = engine.default_params()
    eng.process(torch.zeros(1, 3, 64, 64), p, torch.device("cpu"))
    eng.process(torch.zeros(1, 3, 64, 64), dict(p, seed=7), torch.device("cpu"))
    assert len(calls) == 1


def test_rebuild_on_model_change(monkeypatch):
    eng, calls = make_engine(monkeypatch)
    p = engine.default_params()
    eng.process(torch.zeros(1, 3, 64, 64), p, torch.device("cpu"))
    eng.process(torch.zeros(1, 3, 64, 64), dict(p, model="other/model"), torch.device("cpu"))
    assert len(calls) == 2


def test_first_frame_prepares_before_the_first_call(monkeypatch):
    eng, holder, _ = holding_engine(monkeypatch)
    p = dict(engine.default_params(), prompt="a", strength=0.4, seed=3)
    eng.process(torch.zeros(1, 3, 64, 64), p, torch.device("cpu"))
    w = holder["w"]
    assert w.prepares == [("a", 50)]
    assert w.calls.index("prepare") < w.calls.index("frame")
    assert w.calls.index("update_stream_params") < w.calls.index("frame")
    assert w.stream_params[0]["seed"] == 3
    assert w.stream_params[0]["t_index_list"] == engine.t_indices_for_strength(0.4)
    assert w.frames == 1


def test_prompt_change_reaches_wrapper_once(monkeypatch):
    eng, holder, _ = holding_engine(monkeypatch)
    p = engine.default_params()
    eng.process(torch.zeros(1, 3, 64, 64), dict(p, prompt="a"), torch.device("cpu"))
    eng.process(torch.zeros(1, 3, 64, 64), dict(p, prompt="a"), torch.device("cpu"))
    eng.process(torch.zeros(1, 3, 64, 64), dict(p, prompt="b"), torch.device("cpu"))
    assert holder["w"].prompts.count("b") == 1


def test_prompt_change_does_not_re_prepare_or_touch_stream_params(monkeypatch):
    eng, holder, _ = holding_engine(monkeypatch)
    p = engine.default_params()
    eng.process(torch.zeros(1, 3, 64, 64), dict(p, prompt="a"), torch.device("cpu"))
    w = holder["w"]
    before = len(w.stream_params)
    eng.process(torch.zeros(1, 3, 64, 64), dict(p, prompt="b"), torch.device("cpu"))
    assert len(w.prepares) == 1
    assert len(w.stream_params) == before


def test_strength_and_seed_changes_update_stream_params(monkeypatch):
    eng, holder, _ = holding_engine(monkeypatch)
    p = engine.default_params()
    eng.process(torch.zeros(1, 3, 64, 64), p, torch.device("cpu"))
    eng.process(torch.zeros(1, 3, 64, 64), dict(p, strength=0.9, seed=11), torch.device("cpu"))
    w = holder["w"]
    assert len(w.stream_params) == 2
    assert w.stream_params[-1]["seed"] == 11
    assert w.stream_params[-1]["t_index_list"] == engine.t_indices_for_strength(0.9)
    assert len(w.prepares) == 1


def test_unchanged_params_do_not_re_send_updates(monkeypatch):
    eng, holder, _ = holding_engine(monkeypatch)
    p = engine.default_params()
    for _ in range(3):
        eng.process(torch.zeros(1, 3, 64, 64), p, torch.device("cpu"))
    w = holder["w"]
    assert len(w.prepares) == 1 and len(w.stream_params) == 1 and w.prompts == []
    assert w.frames == 3


def test_build_error_latches_passthrough_and_no_retry(monkeypatch):
    calls = []

    def fake_make(params, device):
        calls.append(1)
        raise RuntimeError("boom on load")

    monkeypatch.setattr(engine, "_make_wrapper", fake_make)
    eng = engine.DiffusionEngine()
    out = torch.full((1, 3, 64, 64), 0.5)
    p = engine.default_params()
    res1 = eng.process(out, p, torch.device("cpu"))
    res2 = eng.process(out, p, torch.device("cpu"))
    assert torch.equal(res1, out) and torch.equal(res2, out)
    assert len(calls) == 1
    assert eng.status.startswith("Error")


def test_error_clears_when_build_key_changes(monkeypatch):
    state = {"fail": True}

    def fake_make(params, device):
        if state["fail"]:
            raise RuntimeError("boom")
        return FakeWrapper()

    monkeypatch.setattr(engine, "_make_wrapper", fake_make)
    eng = engine.DiffusionEngine()
    p = engine.default_params()
    eng.process(torch.zeros(1, 3, 64, 64), p, torch.device("cpu"))
    state["fail"] = False
    res = eng.process(torch.zeros(1, 3, 64, 64), dict(p, model="other/model"), torch.device("cpu"))
    assert res.shape == (1, 3, 512, 512)
    assert eng.status == ""


def test_frame_error_passes_through_without_raising(monkeypatch):
    eng, holder, _ = holding_engine(monkeypatch)
    p = engine.default_params()
    eng.process(torch.zeros(1, 3, 64, 64), p, torch.device("cpu"))
    holder["w"].fail_on_frame = True
    out = torch.full((1, 3, 64, 64), 0.5)
    res = eng.process(out, p, torch.device("cpu"))
    assert torch.equal(res, out)
    assert eng.status.startswith("Error")


def test_stage_output_survives_uint8_conversion(monkeypatch):
    eng, _ = make_engine(monkeypatch)
    out = torch.rand(1, 3, 64, 64) * 2 - 1
    staged = eng.process(out, engine.default_params(), torch.device("cpu"))
    img = staged[0]
    img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0)
    assert img.shape == (512, 512, 3)
    assert img.float().mean() > 10  # not crushed to black
