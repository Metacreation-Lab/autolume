import sys
import time
import types

import pytest
import torch

from diffusion import engine


def pump(eng, out, p, device=torch.device("cpu")):
    """Drive process() until a pending background load settles."""
    res = eng.process(out, p, device)
    deadline = time.time() + 5
    while eng.status.startswith(engine.LOADING_PREFIX) and time.time() < deadline:
        time.sleep(0.001)
        res = eng.process(out, p, device)
    return res


def params(**overrides):
    """Engine params for a test.

    The shipped default has no checkpoint: the panel fills that from the last
    one used or the first one installed, so the engine must be given one here.
    """
    base = dict(engine.default_params(), model="test/model")
    base.update(overrides)
    return base


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
    a = params()
    b = dict(a, prompt="x", strength=0.9, seed=123)
    assert engine.build_key(a) == engine.build_key(b)


def test_build_key_changes_on_model_and_lora():
    a = params()
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
    res = pump(eng, out, params())
    assert res.shape == (1, 3, 512, 512)
    assert res.min() >= -1.0001 and res.max() <= 1.0001
    assert eng.status == ""


def test_no_rebuild_on_prompt_or_strength_change(monkeypatch):
    eng, calls = make_engine(monkeypatch)
    p = params()
    pump(eng, torch.zeros(1, 3, 64, 64), p)
    eng.process(torch.zeros(1, 3, 64, 64), dict(p, prompt="new", strength=0.9), torch.device("cpu"))
    assert len(calls) == 1


def test_no_rebuild_on_seed_change(monkeypatch):
    eng, calls = make_engine(monkeypatch)
    p = params()
    pump(eng, torch.zeros(1, 3, 64, 64), p)
    eng.process(torch.zeros(1, 3, 64, 64), dict(p, seed=7), torch.device("cpu"))
    assert len(calls) == 1


def test_rebuild_on_model_change(monkeypatch):
    eng, calls = make_engine(monkeypatch)
    p = params()
    pump(eng, torch.zeros(1, 3, 64, 64), p)
    pump(eng, torch.zeros(1, 3, 64, 64), dict(p, model="other/model"))
    assert len(calls) == 2


def test_first_frame_prepares_before_the_first_call(monkeypatch):
    eng, holder, _ = holding_engine(monkeypatch)
    p = params(prompt="a", strength=0.4, seed=3)
    pump(eng, torch.zeros(1, 3, 64, 64), p)
    w = holder["w"]
    assert w.prepares == [("a", 50)]
    assert w.calls.index("prepare") < w.calls.index("frame")
    assert w.calls.index("update_stream_params") < w.calls.index("frame")
    assert w.stream_params[0]["seed"] == 3
    assert w.stream_params[0]["t_index_list"] == engine.t_indices_for_strength(0.4)
    assert w.frames == 1


def test_prompt_change_reaches_wrapper_once(monkeypatch):
    eng, holder, _ = holding_engine(monkeypatch)
    p = params()
    pump(eng, torch.zeros(1, 3, 64, 64), dict(p, prompt="a"))
    pump(eng, torch.zeros(1, 3, 64, 64), dict(p, prompt="a"))
    eng.process(torch.zeros(1, 3, 64, 64), dict(p, prompt="b"), torch.device("cpu"))
    assert holder["w"].prompts.count("b") == 1


def test_prompt_change_does_not_re_prepare_or_touch_stream_params(monkeypatch):
    eng, holder, _ = holding_engine(monkeypatch)
    p = params()
    pump(eng, torch.zeros(1, 3, 64, 64), dict(p, prompt="a"))
    w = holder["w"]
    before = len(w.stream_params)
    eng.process(torch.zeros(1, 3, 64, 64), dict(p, prompt="b"), torch.device("cpu"))
    assert len(w.prepares) == 1
    assert len(w.stream_params) == before


def test_strength_and_seed_changes_update_stream_params(monkeypatch):
    eng, holder, _ = holding_engine(monkeypatch)
    p = params()
    pump(eng, torch.zeros(1, 3, 64, 64), p)
    eng.process(torch.zeros(1, 3, 64, 64), dict(p, strength=0.9, seed=11), torch.device("cpu"))
    w = holder["w"]
    assert len(w.stream_params) == 2
    assert w.stream_params[-1]["seed"] == 11
    assert w.stream_params[-1]["t_index_list"] == engine.t_indices_for_strength(0.9)
    assert len(w.prepares) == 1


def test_unchanged_params_do_not_re_send_updates(monkeypatch):
    eng, holder, _ = holding_engine(monkeypatch)
    p = params()
    for _ in range(3):
        pump(eng, torch.zeros(1, 3, 64, 64), p)
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
    p = params()
    res1 = pump(eng, out, p)
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
    p = params()
    pump(eng, torch.zeros(1, 3, 64, 64), p)
    state["fail"] = False
    res = pump(eng, torch.zeros(1, 3, 64, 64), dict(p, model="other/model"))
    assert res.shape == (1, 3, 512, 512)
    assert eng.status == ""


def test_frame_error_passes_through_without_raising(monkeypatch):
    eng, holder, _ = holding_engine(monkeypatch)
    p = params()
    pump(eng, torch.zeros(1, 3, 64, 64), p)
    holder["w"].fail_on_frame = True
    out = torch.full((1, 3, 64, 64), 0.5)
    res = eng.process(out, p, torch.device("cpu"))
    assert torch.equal(res, out)
    assert eng.status.startswith("Error")


def test_lora_params_reach_wrapper(monkeypatch):
    seen = {}

    def fake_make(params, device):
        seen.update(params)
        return FakeWrapper()

    monkeypatch.setattr(engine, "_make_wrapper", fake_make)
    eng = engine.DiffusionEngine()
    p = params(lora_path="/tmp/style.safetensors", lora_scale=0.7)
    pump(eng, torch.zeros(1, 3, 64, 64), p)
    assert seen["lora_path"] == "/tmp/style.safetensors" and seen["lora_scale"] == 0.7


def fake_streamdiffusion_module(monkeypatch, built=None):
    """Stub out the fork so the real _make_wrapper can be exercised off-GPU."""
    module = types.ModuleType("streamdiffusion")

    def constructor(**kwargs):
        if built is not None:
            built.append(kwargs)
        return FakeWrapper()

    module.StreamDiffusionWrapper = constructor
    monkeypatch.setitem(sys.modules, "streamdiffusion", module)


def test_missing_lora_file_errors_without_retry(monkeypatch, tmp_path):
    fake_streamdiffusion_module(monkeypatch)
    calls = []
    real_make = engine._make_wrapper

    def counting_make(params, device):
        calls.append(params["lora_path"])
        return real_make(params, device)

    monkeypatch.setattr(engine, "_make_wrapper", counting_make)
    eng = engine.DiffusionEngine()
    out = torch.full((1, 3, 64, 64), 0.5)
    p = params(lora_path=str(tmp_path / "missing.safetensors"))
    res1 = pump(eng, out, p)
    res2 = eng.process(out, p, torch.device("cpu"))
    assert torch.equal(res1, out) and torch.equal(res2, out)
    assert len(calls) == 1
    assert eng.status.startswith("Error") and "missing.safetensors" in eng.status


def test_a_missing_lora_is_rejected_before_a_tensorrt_build_starts(monkeypatch, tmp_path):
    """The 20 to 30 minute build must not run for a LoRA that is not there.

    The fork swallows a failed LoRA load, so the build would finish and write a
    manifest claiming a LoRA it never fused, poisoning the engine cache under a
    key the user cannot tell apart from a good one.
    """
    import importlib.util
    import os
    import queue
    from diffusion import trt

    monkeypatch.setattr(trt.user_data, "data_path", lambda *parts: tmp_path.joinpath(*parts))
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a, **k: (object() if name == "tensorrt"
                                               else real_find_spec(name, *a, **k)))
    # left unstubbed on purpose: reaching the fork import at all is the failure
    monkeypatch.delitem(sys.modules, "streamdiffusion", raising=False)

    cmd, reply = queue.Queue(), queue.Queue()
    build_params = params(acceleration="tensorrt",
                          lora_path=str(tmp_path / "gone.safetensors"))
    cmd.put({"cmd": "build", "params": build_params})
    trt.run_build(cmd, reply)

    messages = []
    while not reply.empty():
        messages.append(reply.get())
    errors = [m["error"] for m in messages if "error" in m]
    assert errors and "gone.safetensors" in errors[-1]
    assert not any(m.get("done") for m in messages)
    assert not os.path.isdir(trt.engine_dir(build_params))  # no empty dir left behind


def test_lora_error_recovers_when_path_is_fixed(monkeypatch, tmp_path):
    built = []
    fake_streamdiffusion_module(monkeypatch, built)
    eng = engine.DiffusionEngine()
    out = torch.full((1, 3, 64, 64), 0.5)
    p = params()
    pump(eng, out, dict(p, lora_path=str(tmp_path / "missing.safetensors")))
    assert eng.status.startswith("Error")
    assert built == []
    lora = tmp_path / "style.safetensors"
    lora.write_bytes(b"")
    res = pump(eng, out, dict(p, lora_path=str(lora), lora_scale=0.7))
    assert res.shape == (1, 3, 512, 512)
    assert eng.status == ""
    assert built[0]["lora_dict"] == {str(lora): 0.7}


def test_empty_lora_path_builds_without_lora_dict(monkeypatch):
    built = []
    fake_streamdiffusion_module(monkeypatch, built)
    eng = engine.DiffusionEngine()
    pump(eng, torch.zeros(1, 3, 64, 64), params())
    assert eng.status == ""
    assert built[0]["lora_dict"] is None


def test_trt_engine_dir_key_stable_and_sensitive():
    from diffusion import trt
    p = params()
    assert trt.engine_dir_key(p) == trt.engine_dir_key(dict(p, prompt="x", seed=9))
    assert trt.engine_dir_key(p) != trt.engine_dir_key(dict(p, resolution=768))
    assert trt.engine_dir_key(p) != trt.engine_dir_key(dict(p, lora_path="a.safetensors"))


def test_trt_engine_dir_is_a_string_under_the_key():
    from diffusion import trt
    p = params()
    path = trt.engine_dir(p)
    assert isinstance(path, str) and path.endswith(trt.engine_dir_key(p))


def test_trt_engines_ready_needs_the_full_set(monkeypatch, tmp_path):
    from diffusion import trt
    monkeypatch.setattr(trt, "engine_dir", lambda params: str(tmp_path))
    p = params()
    assert not trt.engines_ready(p)
    nested = tmp_path / "stabilityai" / "sd-turbo--mode-img2img"
    nested.mkdir(parents=True)
    (nested / "unet.engine").write_bytes(b"")
    assert not trt.engines_ready(p)
    (nested / "vae_encoder.engine").write_bytes(b"")
    (nested / "vae_decoder.engine").write_bytes(b"")
    assert trt.engines_ready(p)


def trt_params(**kwargs):
    return params(acceleration="tensorrt", **kwargs)


def test_tensorrt_without_engines_runs_unaccelerated(monkeypatch):
    from diffusion import trt
    monkeypatch.setattr(trt, "engines_ready", lambda params: False)
    seen = []

    def fake_make(params, device):
        seen.append(params["acceleration"])
        return FakeWrapper()

    monkeypatch.setattr(engine, "_make_wrapper", fake_make)
    eng = engine.DiffusionEngine()
    res = pump(eng, torch.zeros(1, 3, 64, 64), trt_params())
    # withholding the image would be worse than losing the speed, and the panel
    # flags TensorRT as unbuilt so the loss is visible
    assert res.shape == (1, 3, 512, 512)
    assert seen == ["none"]
    assert eng.status == "" and eng.loaded["acceleration"] == "none"


def test_ticking_tensorrt_without_engines_keeps_the_existing_pipeline(monkeypatch):
    from diffusion import trt
    monkeypatch.setattr(trt, "engines_ready", lambda params: False)
    eng, calls = make_engine(monkeypatch)
    pump(eng, torch.zeros(1, 3, 64, 64), params())
    res = pump(eng, torch.full((1, 3, 64, 64), 0.5), trt_params())
    # same effective setup, so no reload and no dropped frames
    assert len(calls) == 1 and res.shape == (1, 3, 512, 512)


def test_tensorrt_builds_when_engines_are_ready(monkeypatch):
    from diffusion import trt
    monkeypatch.setattr(trt, "engines_ready", lambda params: True)
    eng, calls = make_engine(monkeypatch)
    res = pump(eng, torch.zeros(1, 3, 64, 64), trt_params())
    assert res.shape == (1, 3, 512, 512)
    assert eng.status == "" and len(calls) == 1


def test_engines_appearing_trigger_a_build_without_a_param_change(monkeypatch):
    from diffusion import trt
    state = {"ready": False}
    monkeypatch.setattr(trt, "engines_ready", lambda params: state["ready"])
    eng, calls = make_engine(monkeypatch)
    p = trt_params()
    pump(eng, torch.zeros(1, 3, 64, 64), p)
    assert calls == [engine.build_key(dict(p, acceleration="none"))]
    state["ready"] = True
    res = pump(eng, torch.zeros(1, 3, 64, 64), p)
    # the finished build swaps the unaccelerated pipeline for the compiled one
    assert calls[-1] == engine.build_key(p)
    assert eng.status == "" and res.shape == (1, 3, 512, 512)


def test_tensorrt_wrapper_gets_the_engine_dir_and_never_builds_inline(monkeypatch):
    from diffusion import trt
    monkeypatch.setattr(trt, "engines_ready", lambda params: True)
    built = []
    fake_streamdiffusion_module(monkeypatch, built)
    eng = engine.DiffusionEngine()
    p = trt_params()
    pump(eng, torch.zeros(1, 3, 64, 64), p)
    assert eng.status == ""
    assert built[0]["engine_dir"] == trt.engine_dir(p)
    assert built[0]["build_engines_if_missing"] is False


def test_trt_manifest_roundtrip_lists_the_built_set(monkeypatch, tmp_path):
    import os
    from diffusion import trt
    monkeypatch.setattr(trt.user_data, "data_path", lambda *parts: tmp_path.joinpath(*parts))
    p = trt_params(model="m.safetensors", lora_path="l.safetensors", lora_scale=2.5)
    os.makedirs(trt.engine_dir(p))
    for name in trt.REQUIRED_ENGINES:
        open(os.path.join(trt.engine_dir(p), name), "w").close()
    trt.write_manifest(p)
    assert trt.list_built_engines() == [
        dict(model="m.safetensors", resolution=p["resolution"],
             lora_path="l.safetensors", lora_scale=2.5)]


def test_trt_list_skips_incomplete_stale_and_unmanifested_sets(monkeypatch, tmp_path):
    import json
    import os
    from diffusion import trt
    monkeypatch.setattr(trt.user_data, "data_path", lambda *parts: tmp_path.joinpath(*parts))
    manifest_only = trt_params(model="manifest-only")
    os.makedirs(trt.engine_dir(manifest_only))
    trt.write_manifest(manifest_only)
    stale = trt_params(model="stale-fork")
    os.makedirs(trt.engine_dir(stale))
    for name in trt.REQUIRED_ENGINES:
        open(os.path.join(trt.engine_dir(stale), name), "w").close()
    trt.write_manifest(stale)
    manifest_path = os.path.join(trt.engine_dir(stale), "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    manifest["fork_sha"] = "old"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    engines_only = trt_params(model="engines-only")
    os.makedirs(trt.engine_dir(engines_only))
    for name in trt.REQUIRED_ENGINES:
        open(os.path.join(trt.engine_dir(engines_only), name), "w").close()
    assert trt.list_built_engines() == []


def test_load_does_not_block_and_reports_progress(monkeypatch):
    import threading
    release = threading.Event()

    def slow_make(params, device):
        release.wait(5)
        return FakeWrapper()

    monkeypatch.setattr(engine, "_make_wrapper", slow_make)
    eng = engine.DiffusionEngine()
    out = torch.full((1, 3, 64, 64), 0.5)
    p = params()
    for _ in range(3):
        res = eng.process(out, p, torch.device("cpu"))
        assert torch.equal(res, out)  # frames keep flowing, undiffused
        assert eng.status.startswith(engine.LOADING_PREFIX)
    release.set()
    assert pump(eng, out, p).shape == (1, 3, 512, 512)
    assert eng.status == ""


def test_loading_flag_tracks_the_background_load(monkeypatch):
    import threading
    release = threading.Event()

    def slow_make(params, device):
        release.wait(5)
        return FakeWrapper()

    monkeypatch.setattr(engine, "_make_wrapper", slow_make)
    eng = engine.DiffusionEngine()
    out = torch.zeros(1, 3, 64, 64)
    p = params()
    assert eng.loading is False
    eng.process(out, p, torch.device("cpu"))
    # the render loop keeps producing frames while this is set, otherwise the
    # finished pipeline would never be installed on an idle scene
    assert eng.loading is True
    release.set()
    pump(eng, out, p)
    assert eng.loading is False and eng.status == ""


def test_a_failed_load_is_logged_with_its_traceback(monkeypatch, caplog):
    """Without this a load that failed is indistinguishable in the log from one
    that was never attempted, which is exactly what a diagnosis needs to tell
    apart."""
    def failing_make(params, device):
        raise RuntimeError("no such checkpoint")

    monkeypatch.setattr(engine, "_make_wrapper", failing_make)
    eng = engine.DiffusionEngine()
    with caplog.at_level("ERROR"):
        pump(eng, torch.zeros(1, 3, 64, 64), params())
    assert "Diffusion pipeline failed to load" in caplog.text
    assert "no such checkpoint" in caplog.text  # the traceback, not just a label


def test_a_failing_frame_is_logged_once_not_every_frame(monkeypatch, caplog):
    eng, holder, _calls = holding_engine(monkeypatch)
    out = torch.zeros(1, 3, 64, 64)
    p = params()
    pump(eng, out, p)
    holder["w"].fail_on_frame = True
    with caplog.at_level("ERROR"):
        for _ in range(5):
            eng.process(out, p, torch.device("cpu"))
    # a frame fails at frame rate: one entry, not one per frame
    assert caplog.text.count("Diffusion frame failed") == 1
    assert "boom on frame" in caplog.text


def test_smoothing_off_returns_the_frame_untouched(monkeypatch):
    eng, holder, _calls = holding_engine(monkeypatch)
    out = torch.zeros(1, 3, 64, 64)
    p = params(smoothing=0.0)
    first = pump(eng, out, p).clone()
    second = eng.process(out, p, torch.device("cpu"))
    # the fake wrapper returns fresh noise every frame, so an untouched second
    # frame must not resemble the first
    assert not torch.allclose(first, second)


def test_smoothing_pulls_each_frame_towards_the_last(monkeypatch):
    eng, _holder, _calls = holding_engine(monkeypatch)
    out = torch.zeros(1, 3, 64, 64)
    p = params(smoothing=0.8)
    first = pump(eng, out, p).clone()
    second = eng.process(out, p, torch.device("cpu"))
    # 0.8 of the previous frame survives, so consecutive frames move less than
    # the raw pipeline output does
    moved = (second - first).abs().mean()
    eng._previous = None
    raw_a = eng.process(out, dict(p, smoothing=0.0), torch.device("cpu")).clone()
    raw_b = eng.process(out, dict(p, smoothing=0.0), torch.device("cpu"))
    assert moved < (raw_b - raw_a).abs().mean()


def test_smoothing_never_freezes_the_output(monkeypatch):
    eng, _holder, _calls = holding_engine(monkeypatch)
    out = torch.zeros(1, 3, 64, 64)
    p = params(smoothing=1.0)  # clamped below 1
    first = pump(eng, out, p).clone()
    second = eng.process(out, p, torch.device("cpu"))
    assert not torch.equal(first, second)


def test_smoothing_is_a_live_control_not_a_rebuild(monkeypatch):
    eng, calls = make_engine(monkeypatch)
    p = params()
    pump(eng, torch.zeros(1, 3, 64, 64), p)
    eng.process(torch.zeros(1, 3, 64, 64), dict(p, smoothing=0.7), torch.device("cpu"))
    assert len(calls) == 1
    assert engine.build_key(p) == engine.build_key(dict(p, smoothing=0.7))


def test_smoothing_history_is_dropped_when_the_pipeline_changes(monkeypatch):
    eng, _holder, _calls = holding_engine(monkeypatch)
    out = torch.zeros(1, 3, 64, 64)
    p = params(smoothing=0.8)
    pump(eng, out, p)
    assert eng._previous is not None
    # a frame from the old pipeline must not bleed into the new one
    eng.process(out, dict(p, model="other/model"), torch.device("cpu"))
    assert eng._previous is None


def test_smoothing_survives_a_resolution_change(monkeypatch):
    eng, _holder, _calls = holding_engine(monkeypatch)
    out = torch.zeros(1, 3, 64, 64)
    p = params(smoothing=0.8)
    pump(eng, out, p)
    res = pump(eng, out, dict(p, resolution=768))
    assert res.shape == (1, 3, 768, 768)  # no blend against a mismatched shape


def test_loaded_reports_what_is_actually_in_vram(monkeypatch):
    eng, _calls = make_engine(monkeypatch)
    out = torch.zeros(1, 3, 64, 64)
    p = params(lora_path="age.safetensors", lora_scale=3.0)
    assert eng.loaded is None
    pump(eng, out, p)
    # the UI reports checkpoint and LoRA separately, so it needs the live params
    assert eng.loaded["model"] == p["model"]
    assert eng.loaded["lora_path"] == "age.safetensors"
    assert eng.loaded["lora_scale"] == 3.0


def test_loaded_clears_the_moment_the_setup_changes(monkeypatch):
    import threading
    release = threading.Event()

    def slow_make(params, device):
        release.wait(5)
        return FakeWrapper()

    monkeypatch.setattr(engine, "_make_wrapper", slow_make)
    eng = engine.DiffusionEngine()
    out = torch.zeros(1, 3, 64, 64)
    p = params()
    release.set()
    pump(eng, out, p)
    assert eng.loaded is not None
    release.clear()
    eng.process(out, dict(p, model="other/model"), torch.device("cpu"))
    # the old pipeline is gone and the new one is not up: nothing is live
    assert eng.loaded is None
    release.set()


def test_loaded_stays_none_when_the_load_fails(monkeypatch):
    def failing_make(params, device):
        raise RuntimeError("no such checkpoint")

    monkeypatch.setattr(engine, "_make_wrapper", failing_make)
    eng = engine.DiffusionEngine()
    pump(eng, torch.zeros(1, 3, 64, 64), params())
    assert eng.loaded is None and eng.status.startswith("Error")


def test_load_started_for_an_abandoned_key_is_discarded(monkeypatch):
    import threading
    release = threading.Event()
    made = []

    def slow_make(params, device):
        made.append(params["model"])
        release.wait(5)
        return FakeWrapper()

    monkeypatch.setattr(engine, "_make_wrapper", slow_make)
    eng = engine.DiffusionEngine()
    out = torch.zeros(1, 3, 64, 64)
    p = params()
    eng.process(out, p, torch.device("cpu"))
    other = dict(p, model="other/model")
    eng.process(out, other, torch.device("cpu"))  # user moves on mid-load
    release.set()
    pump(eng, out, other)
    assert eng.status == "" and made[-1] == "other/model"


def test_only_one_load_runs_per_key_change(monkeypatch):
    import threading
    release = threading.Event()
    made = []

    def slow_make(params, device):
        made.append(1)
        release.wait(5)
        return FakeWrapper()

    monkeypatch.setattr(engine, "_make_wrapper", slow_make)
    eng = engine.DiffusionEngine()
    out = torch.zeros(1, 3, 64, 64)
    p = params()
    for _ in range(5):
        eng.process(out, p, torch.device("cpu"))
    release.set()
    pump(eng, out, p)
    assert len(made) == 1


def test_build_progress_tracks_artifacts_on_disk(tmp_path):
    import os
    from diffusion import trt
    root = str(tmp_path)
    assert trt.build_progress(root, 0) == "Exporting unet (1 of 3)"
    open(os.path.join(root, "unet.onnx"), "w").close()
    assert trt.build_progress(root, 0).startswith("Compiling unet (1 of 3)")
    open(os.path.join(root, "unet.engine"), "w").close()
    assert trt.build_progress(root, 0).startswith("Exporting vae_decoder (2 of 3)")
    for name in trt.REQUIRED_ENGINES:
        open(os.path.join(root, name), "w").close()
    assert trt.build_progress(root, 0).startswith("Finishing")
    assert "5 min elapsed" in trt.build_progress(root, 5 * 60 + 3)


def test_build_fraction_is_monotonic_across_stages():
    from widgets.diffusion_widget import build_fraction
    steps = [build_fraction(m) for m in
             ["Exporting unet (1 of 3)", "Compiling unet (1 of 3)",
              "Exporting vae_decoder (2 of 3)", "Compiling vae_decoder (2 of 3)",
              "Exporting vae_encoder (3 of 3)", "Finishing"]]
    assert steps == sorted(steps)
    assert steps[0] == 0.0 and steps[-1] == 1.0


def test_stage_output_survives_uint8_conversion(monkeypatch):
    eng, _ = make_engine(monkeypatch)
    out = torch.rand(1, 3, 64, 64) * 2 - 1
    staged = pump(eng, out, params())
    img = staged[0]
    img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0)
    assert img.shape == (512, 512, 3)
    assert img.float().mean() > 10  # not crushed to black
