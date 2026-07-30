import logging
import threading
import time

import pytest

from autolume.live.core.generator import (
    DeviceStatus,
    DeviceUnavailable,
    LayerInfo,
    LoadedModel,
    MixSaveStatus,
    ModelHost,
    ModelInfo,
    adjust_weights,
    channel_window,
    corner_seeds,
    derive_float_image,
    direction_delta,
    effective_noise_seed,
    manipulation_dict,
    noise_mode,
    resolve_device,
    slerp,
    to_uint8_frame,
    usable_indices,
)
from autolume.live.core.generator import _LOG_ONCE_CAP
from autolume.live.core.mixing import (
    INCOMPATIBLE_MODELS,
    conv_names,
    layer_resolution,
    selection_length,
)
from autolume.live.core import presets
from autolume.live.core.events import ControlEvent
from autolume.live.core.mapping import apply_event
from autolume.live.core.params import (
    BEND_RATIO,
    ControlState,
    Keyframe,
    SetLayerRatio,
    Transform,
    to_render_params,
)


# The bending tests below make the generator import the operator library,
# and merely importing kornia trips a torch FutureWarning from its lightglue
# submodule. Matched by message so it cannot mask anything else. The mixing
# tests at the end build real generators, whose import chain warns once about
# pkg_resources.
pytestmark = [
    pytest.mark.filterwarnings(
        r"ignore:.*torch\.cuda\.amp\.custom_fwd.*:FutureWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:pkg_resources is deprecated.*:DeprecationWarning"
    ),
]


def render_params(**changes):
    return to_render_params(ControlState(**changes))


def test_corner_seeds_integer_position_is_single_seed():
    corners = corner_seeds(3.0, 2.0, step_y=100)
    assert corners == [(3 + 2 * 100, 1.0)]


def test_corner_seeds_weights_sum_to_one():
    corners = corner_seeds(1.25, 7.5, step_y=100)
    assert len(corners) == 4
    assert abs(sum(w for _, w in corners) - 1.0) < 1e-9


def test_corner_seeds_negative_positions_wrap_to_uint32():
    corners = corner_seeds(-1.5, 0.0, step_y=100)
    assert all(0 <= seed < 2**32 for seed, _ in corners)


def test_corner_seeds_deterministic():
    assert corner_seeds(0.3, 0.7) == corner_seeds(0.3, 0.7)


class FakeModel:
    def __init__(self, path):
        self.pkl_path = path
        self.z_dim = 4
        self.num_ws = 2


def test_model_host_loads_in_background():
    host = ModelHost(loader=FakeModel)
    assert host.current() is None
    host.request_load("/tmp/a.pkl")
    deadline = time.monotonic() + 2.0
    while host.current() is None and time.monotonic() < deadline:
        time.sleep(0.005)
    assert host.current().pkl_path == "/tmp/a.pkl"
    assert host.error() is None
    host.stop()


def test_model_host_surfaces_loader_error():
    def failing(path):
        raise RuntimeError("bad pkl")

    host = ModelHost(loader=failing)
    host.request_load("/tmp/bad.pkl")
    deadline = time.monotonic() + 2.0
    while host.error() is None and time.monotonic() < deadline:
        time.sleep(0.005)
    assert "bad pkl" in host.error()
    assert host.current() is None
    host.stop()


def test_model_host_coalesces_to_newest_request():
    release = threading.Event()
    loaded = []

    def slow(path):
        release.wait(timeout=2.0)
        loaded.append(path)
        return FakeModel(path)

    host = ModelHost(loader=slow)
    host.request_load("/tmp/a.pkl")
    host.request_load("/tmp/b.pkl")
    host.request_load("/tmp/c.pkl")
    release.set()
    deadline = time.monotonic() + 2.0
    while (host.current() is None or host.current().pkl_path != "/tmp/c.pkl") and \
            time.monotonic() < deadline:
        time.sleep(0.005)
    assert host.current().pkl_path == "/tmp/c.pkl"
    assert "/tmp/b.pkl" not in loaded
    host.stop()


def test_model_host_publishes_model_info_on_successful_load():
    host = ModelHost(loader=FakeModel)
    assert host.info_store.snapshot() is None
    host.request_load("/tmp/a.pkl")
    deadline = time.monotonic() + 2.0
    while host.info_store.snapshot() is None and time.monotonic() < deadline:
        time.sleep(0.005)
    assert host.info_store.snapshot() == ModelInfo(
        pkl_path="/tmp/a.pkl", z_dim=4, num_ws=2
    )
    host.stop()


def test_model_host_keeps_model_info_when_a_load_fails_mid_set():
    """I2: a failed load leaves the previous model rendering, so its catalog
    must stay published.

    This used to pin the opposite: `info_store.set(None)` on failure, while
    `current()` kept returning the live model. That mismatch froze the noise
    loop, turned Randomize into a no-op about "no model" and collapsed the
    bending panel, all while the picture kept rendering, with no recovery
    short of a successful load. `error()` is the channel that carries the
    failure; the catalog belongs to whatever is actually on screen.
    """
    should_fail = threading.Event()

    def loader(path):
        if should_fail.is_set():
            raise RuntimeError("bad pkl")
        return FakeModel(path)

    host = ModelHost(loader=loader)
    host.request_load("/tmp/a.pkl")
    deadline = time.monotonic() + 2.0
    while host.info_store.snapshot() is None and time.monotonic() < deadline:
        time.sleep(0.005)
    assert host.info_store.snapshot() is not None

    should_fail.set()
    host.request_load("/tmp/bad.pkl")
    deadline = time.monotonic() + 2.0
    while host.error() is None and time.monotonic() < deadline:
        time.sleep(0.005)

    assert host.error() is not None
    assert host.current().pkl_path == "/tmp/a.pkl"
    assert host.info_store.snapshot() == ModelInfo(
        pkl_path="/tmp/a.pkl", z_dim=4, num_ws=2
    )
    host.stop()


# --- device switching ----------------------------------------------------


def test_resolve_device_auto_delegates_to_pick_device(monkeypatch):
    import autolume.live.core.generator as generator_module

    sentinel = object()
    monkeypatch.setattr(generator_module, "pick_device", lambda: sentinel)
    assert resolve_device("auto") is sentinel


def test_resolve_device_cpu_is_always_available():
    import torch

    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_cuda_unavailable_raises(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(DeviceUnavailable):
        resolve_device("cuda")


def test_resolve_device_mps_unavailable_raises(monkeypatch):
    import torch

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(DeviceUnavailable):
        resolve_device("mps")


def test_resolve_device_unknown_name_raises():
    with pytest.raises(DeviceUnavailable):
        resolve_device("tpu")


class _DeviceAwareModel:
    """A loader double that records the device it was built with and can be
    released, so tests can assert `ModelHost` retires it properly."""

    def __init__(self, path, device=None):
        self.pkl_path = path
        self.z_dim = 4
        self.num_ws = 2
        self.device = device
        self.released = 0

    def release(self) -> None:
        self.released += 1


def test_model_host_request_device_is_a_noop_without_a_current_model():
    calls = []

    def loader(path, device=None):
        calls.append((path, device))
        return _DeviceAwareModel(path, device)

    host = ModelHost(loader=loader)
    host.request_device("cpu")
    time.sleep(0.05)
    assert host.current() is None
    assert calls == []
    host.stop()


def test_model_host_reloads_the_current_pkl_onto_a_resolved_device():
    import torch

    calls = []

    def loader(path, device=None):
        calls.append((path, device))
        return _DeviceAwareModel(path, device)

    host = ModelHost(loader=loader)
    host.request_load("/tmp/a.pkl")
    deadline = time.monotonic() + 2.0
    while host.current() is None and time.monotonic() < deadline:
        time.sleep(0.005)
    first = host.current()
    assert calls == [("/tmp/a.pkl", None)]

    host.request_device("cpu")
    deadline = time.monotonic() + 2.0
    while host.current() is first and time.monotonic() < deadline:
        time.sleep(0.005)

    assert calls[-1] == ("/tmp/a.pkl", torch.device("cpu"))
    assert host.current().device == torch.device("cpu")
    assert host.device_store.snapshot() == DeviceStatus(
        active="cpu", requested="cpu", error=None
    )
    # The retired model is released the instant it stops being current, not
    # whenever GC gets to it (see LoadedModel.release / the hook cycle note).
    assert first.released == 1
    host.stop()


def test_model_host_reverts_status_and_keeps_the_model_on_an_unavailable_device():
    def loader(path, device=None):
        return _DeviceAwareModel(path, device)

    host = ModelHost(loader=loader)
    host.request_load("/tmp/a.pkl")
    deadline = time.monotonic() + 2.0
    while host.current() is None and time.monotonic() < deadline:
        time.sleep(0.005)
    running = host.current()

    # This machine (the test runner) never has CUDA, so this is a genuine
    # unavailable-device request, not a mocked one.
    host.request_device("cuda")
    deadline = time.monotonic() + 2.0
    while host.device_store.snapshot().error is None and time.monotonic() < deadline:
        time.sleep(0.005)

    assert host.current() is running
    status = host.device_store.snapshot()
    assert status.requested == "cuda"
    assert status.error is not None
    assert running.released == 0
    host.stop()


def test_model_host_a_reload_failure_on_device_switch_also_reverts():
    def loader(path, device=None):
        if device is not None:
            raise RuntimeError("out of memory")
        return _DeviceAwareModel(path, device)

    host = ModelHost(loader=loader)
    host.request_load("/tmp/a.pkl")
    deadline = time.monotonic() + 2.0
    while host.current() is None and time.monotonic() < deadline:
        time.sleep(0.005)
    running = host.current()

    host.request_device("cpu")
    deadline = time.monotonic() + 2.0
    while host.device_store.snapshot().error is None and time.monotonic() < deadline:
        time.sleep(0.005)

    assert host.current() is running
    assert "out of memory" in host.device_store.snapshot().error
    host.stop()


def test_model_host_releases_the_outgoing_model_on_a_plain_reload():
    def loader(path):
        return _DeviceAwareModel(path)

    host = ModelHost(loader=loader)
    host.request_load("/tmp/a.pkl")
    deadline = time.monotonic() + 2.0
    while host.current() is None and time.monotonic() < deadline:
        time.sleep(0.005)
    first = host.current()

    host.request_load("/tmp/b.pkl")
    deadline = time.monotonic() + 2.0
    while host.current() is first and time.monotonic() < deadline:
        time.sleep(0.005)

    assert host.current().pkl_path == "/tmp/b.pkl"
    assert first.released == 1
    host.stop()


def test_model_host_a_later_plain_load_keeps_the_selected_device():
    """Regression: `_load_default` used to ignore any earlier device switch
    and let the loader's own default (pick_device()) choose again, so the
    state kept saying "cpu" while a brand new model quietly loaded
    elsewhere."""
    import torch

    calls = []

    def loader(path, device=None):
        calls.append((path, device))
        return _DeviceAwareModel(path, device)

    host = ModelHost(loader=loader)
    host.request_load("/tmp/a.pkl")
    deadline = time.monotonic() + 2.0
    while host.current() is None and time.monotonic() < deadline:
        time.sleep(0.005)

    host.request_device("cpu")
    deadline = time.monotonic() + 2.0
    while (
        host.current().device != torch.device("cpu")
        and time.monotonic() < deadline
    ):
        time.sleep(0.005)
    assert host.current().device == torch.device("cpu")

    host.request_load("/tmp/b.pkl")

    def loaded_b():
        current = host.current()
        return current is not None and current.pkl_path == "/tmp/b.pkl"

    deadline = time.monotonic() + 2.0
    while not loaded_b() and time.monotonic() < deadline:
        time.sleep(0.005)

    assert loaded_b()
    assert host.current().device == torch.device("cpu")
    assert calls[-1] == ("/tmp/b.pkl", torch.device("cpu"))
    assert host.device_store.snapshot() == DeviceStatus(
        active="cpu", requested="cpu", error=None
    )
    host.stop()


def test_model_host_a_device_request_before_any_model_is_honored_on_first_load():
    """Regression: request_device used to be a pure no-op with nothing
    loaded, so a device picked before the first pkl was silently dropped
    the instant that pkl actually loaded."""
    import torch

    calls = []

    def loader(path, device=None):
        calls.append((path, device))
        return _DeviceAwareModel(path, device)

    host = ModelHost(loader=loader)
    host.request_device("cpu")
    assert host.current() is None
    time.sleep(0.05)
    assert calls == []

    host.request_load("/tmp/a.pkl")
    deadline = time.monotonic() + 2.0
    while host.current() is None and time.monotonic() < deadline:
        time.sleep(0.005)

    assert calls == [("/tmp/a.pkl", torch.device("cpu"))]
    assert host.current().device == torch.device("cpu")
    assert host.device_store.snapshot() == DeviceStatus(
        active="cpu", requested="cpu", error=None
    )
    host.stop()


def test_model_host_an_unavailable_device_picked_before_any_model_still_loads_it():
    """Regression: `_load_default`'s own DeviceUnavailable branch cleared
    `_pending` unconditionally, so picking an unavailable device before
    ever loading a model dropped that first `request_load` silently:
    `current()` stayed None, `error()` reported the device failure, but
    nothing retried, and because `_ModelWatchingControlLoop` is edge
    triggered on `pkl_path`, re-picking the very same path from the UI did
    nothing either. `_load_on_device` already had this fix; `_load_default`
    now matches it: `_device_name` is restored to a name that resolves and
    `_pending` stays put, so the loader thread's own re-wake retries the
    same pkl immediately."""

    def loader(path, device=None):
        return _DeviceAwareModel(path, device)

    host = ModelHost(loader=loader)

    # This machine never has CUDA, so this genuinely fails, before any pkl
    # has ever been requested.
    host.request_device("cuda")
    host.request_load("/tmp/a.pkl")

    deadline = time.monotonic() + 2.0
    while host.current() is None and time.monotonic() < deadline:
        time.sleep(0.005)

    assert host.current() is not None
    assert host.current().pkl_path == "/tmp/a.pkl"
    assert host.current().device is None  # "auto": the loader's own choice
    host.stop()


def test_model_host_a_device_that_never_resolves_does_not_spin(monkeypatch):
    """Regression: the retry `_load_default` performs after a device
    resolution failure trusted an unstated invariant, that the device
    name restored from `_current.device` always resolves. A `.device`
    whose string form `resolve_device` never matches broke that
    invariant and spun the loader thread forever: every iteration took
    the lock, called `resolve_device`, logged a warning and re-set the
    wakeup, with no backoff, before this fix bounded it to one retry per
    path. An iteration count on `resolve_device`, not a timing
    measurement, is the honest way to pin "does not spin"."""
    import autolume.live.core.generator as generator_module

    class _UnresolvableDevice:
        def __str__(self):
            return "quantum"

    real_resolve_device = generator_module.resolve_device
    resolve_calls = []

    def counting_resolve_device(name):
        resolve_calls.append(name)
        return real_resolve_device(name)

    monkeypatch.setattr(generator_module, "resolve_device", counting_resolve_device)

    def loader(path, device=None):
        model = _DeviceAwareModel(path, device)
        model.device = _UnresolvableDevice()
        return model

    host = ModelHost(loader=loader)
    host.request_load("/tmp/a.pkl")
    deadline = time.monotonic() + 2.0
    while host.current() is None and time.monotonic() < deadline:
        time.sleep(0.005)
    assert host.current() is not None

    # This machine never has CUDA, so this genuinely fails. a.pkl is
    # already current, so _load_on_device's own already_current
    # shortcut takes it (no retry there); it still restores the sticky
    # device name to the unresolvable value read off a.pkl's `.device`.
    host.request_device("cuda")
    deadline = time.monotonic() + 2.0
    while host.device_store.snapshot().error is None and time.monotonic() < deadline:
        time.sleep(0.005)

    resolve_calls.clear()
    host.request_load("/tmp/b.pkl")
    deadline = time.monotonic() + 2.0
    while host.error() is None and time.monotonic() < deadline:
        time.sleep(0.005)

    assert host.error() is not None
    assert host.current().pkl_path == "/tmp/a.pkl"  # b.pkl never loaded
    # a.pkl keeps rendering, so its catalog stays published: giving up on
    # b.pkl must not clear the info for the model still on screen (I2).
    assert host.info_store.snapshot() is not None
    assert host.info_store.snapshot().pkl_path == "/tmp/a.pkl"
    # Bounded: one attempt, one retry, then stop. Never unbounded.
    assert resolve_calls.count("quantum") == 2
    host.stop()


def test_model_host_a_plain_load_for_the_already_current_pkl_is_a_no_op_on_device_failure():
    """Regression: `_load_on_device` short circuits when the path being
    redirected is already `_current` (nothing to reload). `_load_default`
    had no equivalent, so a plain `request_load` for the pkl already
    loaded, hitting a transiently bad sticky device, triggered a full
    redundant reload and swap instead of doing nothing, unlike the
    sibling branch.

    `_device_name` is set directly here rather than through
    `request_device`: going through it would route through
    `_load_on_device`'s own `already_current` shortcut first and correct
    `_device_name` before `_load_default` ever saw a bad one, which is
    exactly why this needs its own fix rather than inheriting that one.
    """

    def loader(path, device=None):
        calls.append((path, device))
        return _DeviceAwareModel(path, device)

    calls = []
    host = ModelHost(loader=loader)
    host.request_load("/tmp/a.pkl")
    deadline = time.monotonic() + 2.0
    while host.current() is None and time.monotonic() < deadline:
        time.sleep(0.005)
    first = host.current()

    calls.clear()
    # This machine never has CUDA, so this genuinely fails.
    host._device_name = "cuda"
    host.request_load("/tmp/a.pkl")
    time.sleep(0.1)

    assert calls == []
    assert host.current() is first
    host.stop()


def test_model_host_a_device_request_during_an_in_flight_pkl_load_does_not_strand_it():
    """Regression: request_device used to overwrite `_pending` with
    whatever was already `current()`, so a pkl load already in flight lost
    the coalescing race to a reload of the *old* model. The new pkl never
    won, and re-picking it did nothing because the control side already
    believed it was current."""
    import torch

    release = threading.Event()
    seen = []

    def loader(path, device=None):
        seen.append((path, device))
        if path == "/tmp/b.pkl" and device is None:
            # Only the first, device-less attempt at b.pkl blocks; the
            # device-redirected retry (once request_device fires) sails
            # through immediately, the way a fast reload would.
            release.wait(timeout=2.0)
        return _DeviceAwareModel(path, device)

    host = ModelHost(loader=loader)
    host.request_load("/tmp/a.pkl")
    deadline = time.monotonic() + 2.0
    while host.current() is None and time.monotonic() < deadline:
        time.sleep(0.005)

    host.request_load("/tmp/b.pkl")
    deadline = time.monotonic() + 2.0
    while ("/tmp/b.pkl", None) not in seen and time.monotonic() < deadline:
        time.sleep(0.005)

    host.request_device("cpu")
    release.set()

    def loaded_b_on_cpu():
        current = host.current()
        return (
            current is not None
            and current.pkl_path == "/tmp/b.pkl"
            and current.device == torch.device("cpu")
        )

    deadline = time.monotonic() + 2.0
    while not loaded_b_on_cpu() and time.monotonic() < deadline:
        time.sleep(0.005)

    assert loaded_b_on_cpu()
    host.stop()


def test_model_host_a_failed_switch_does_not_strand_the_sticky_device():
    """Regression: request_device set _device_name unconditionally and
    nothing ever restored it on failure, so every plain load after a
    failed switch kept hitting the same dead device forever, silently
    (no error(), no info_store, `_pending` just cleared)."""
    import torch

    calls = []

    def loader(path, device=None):
        # Resolves the auto/no-device case to a fixed concrete device, the
        # way a real loader's own `device or pick_device()` would, so
        # `.device` reads a comparable, deterministic value regardless of
        # whether this call came through the "auto" path or an explicit one.
        resolved = device or torch.device("cpu")
        calls.append((path, device))
        return _DeviceAwareModel(path, resolved)

    host = ModelHost(loader=loader)
    host.request_load("/tmp/a.pkl")
    deadline = time.monotonic() + 2.0
    while host.current() is None and time.monotonic() < deadline:
        time.sleep(0.005)

    # This machine never has CUDA, so this genuinely fails.
    host.request_device("cuda")
    deadline = time.monotonic() + 2.0
    while host.device_store.snapshot().error is None and time.monotonic() < deadline:
        time.sleep(0.005)
    assert host.device_store.snapshot().error is not None

    host.request_load("/tmp/b.pkl")

    def loaded_b():
        current = host.current()
        return current is not None and current.pkl_path == "/tmp/b.pkl"

    deadline = time.monotonic() + 2.0
    while not loaded_b() and time.monotonic() < deadline:
        time.sleep(0.005)

    assert loaded_b()
    assert host.current().device == torch.device("cpu")
    assert host.error() is None
    assert host.info_store.snapshot() is not None
    # The sticky device was restored to "cpu" (read straight off a.pkl's
    # own device) rather than dropped back to "auto", so b.pkl resolves it
    # explicitly here rather than falling through to a bare loader(path).
    assert calls[-1] == ("/tmp/b.pkl", torch.device("cpu"))
    host.stop()


def test_model_host_a_failed_switch_during_an_in_flight_load_does_not_drop_the_pkl():
    """Regression: request_device redirecting a pkl load already in flight
    (see the test above this one) used to be fine only because that
    redirected reload always succeeded. When it fails, the old code
    cleared both `_pending` and `_pending_device` and gave up: the pkl the
    user asked for was never loaded, never reported as failed either
    (`error()` stayed None, `loading()` stayed False)."""
    import torch

    release = threading.Event()

    def loader(path, device=None):
        if path == "/tmp/b.pkl" and device is None:
            release.wait(timeout=2.0)
        if device is not None and str(device) == "cuda":
            raise RuntimeError("should never reach the loader for cuda here")
        resolved = device or torch.device("cpu")
        return _DeviceAwareModel(path, resolved)

    host = ModelHost(loader=loader)
    host.request_load("/tmp/a.pkl")
    deadline = time.monotonic() + 2.0
    while host.current() is None and time.monotonic() < deadline:
        time.sleep(0.005)

    host.request_load("/tmp/b.pkl")
    time.sleep(0.05)  # give the loader thread time to enter the block on b.pkl

    # This machine never has CUDA: resolve_device fails before the loader
    # is ever called for "cuda", so b.pkl's load is redirected and then
    # immediately fails to resolve, never touching the loader with device.
    host.request_device("cuda")
    release.set()

    def loaded_b():
        current = host.current()
        return current is not None and current.pkl_path == "/tmp/b.pkl"

    deadline = time.monotonic() + 2.0
    while not loaded_b() and time.monotonic() < deadline:
        time.sleep(0.005)

    assert loaded_b()
    assert host.current().device == torch.device("cpu")
    host.stop()


class _FakeMapping:
    def __init__(self, w_avg, num_ws):
        self.w_avg = w_avg
        self.c_dim = 0
        self._num_ws = num_ws

    def __call__(self, z, c, truncation_psi):
        import torch

        return torch.zeros([z.shape[0], self._num_ws, self.w_avg.shape[0]])


class _FakeG:
    z_dim = 4
    num_ws = 2

    def __init__(self, synthesis):
        import torch

        self.mapping = _FakeMapping(torch.zeros([8]), num_ws=2)
        self.synthesis = synthesis


def _fake_model(synthesis):
    import torch

    from autolume.live.core.generator import LoadedModel

    return LoadedModel("/tmp/fake.pkl", _FakeG(synthesis), torch.device("cpu"))


def _zeros_synthesis(ws, noise_mode):
    import torch

    return torch.zeros([1, 3, 8, 8])


def test_render_frame_with_tensor_synthesis_output():
    model = _fake_model(_zeros_synthesis)
    frame = model.render_frame(render_params(), 0)
    assert frame.shape == (8, 8, 3)
    assert frame.dtype.name == "uint8"


def test_render_frame_with_tuple_synthesis_output():
    def synthesis(ws, noise_mode):
        import torch

        return torch.zeros([1, 3, 8, 8]), []

    model = _fake_model(synthesis)
    frame = model.render_frame(render_params(), 0)
    assert frame.shape == (8, 8, 3)


@pytest.mark.parametrize(
    "changes,expected",
    [
        ({"noise_enabled": False}, "none"),
        ({"noise_enabled": False, "noise_seed": 9, "noise_anim": True}, "none"),
        ({"noise_seed": 0, "noise_anim": False}, "const"),
        ({"noise_seed": 9, "noise_anim": False}, "random"),
        ({"noise_seed": 0, "noise_anim": True}, "random"),
        # A layer ratio does not change the mode: the synthesis layer draws its
        # random field on the ratio adjusted grid, so animated noise and a
        # ratio work together.
        (
            {"noise_anim": True, "layer_ratios": (("conv1", 2.0, 1.0),)},
            "random",
        ),
        (
            {"noise_seed": 9, "layer_ratios": (("conv1", 1.0, 0.5),)},
            "random",
        ),
        (
            {
                "noise_enabled": False,
                "noise_anim": True,
                "layer_ratios": (("conv1", 2.0, 1.0),),
            },
            "none",
        ),
    ],
)
def test_noise_mode_truth_table(changes, expected):
    assert noise_mode(render_params(**changes)) == expected


def test_effective_noise_seed_static_ignores_frame_index():
    params = render_params(noise_seed=42, noise_anim=False)
    assert effective_noise_seed(params, 0) == 42
    assert effective_noise_seed(params, 137) == 42


def test_effective_noise_seed_animated_advances_with_frame_index():
    params = render_params(noise_seed=42, noise_anim=True)
    assert effective_noise_seed(params, 0) == 42
    assert effective_noise_seed(params, 3) == 45


def test_effective_noise_seed_stays_within_32_bits():
    params = render_params(noise_seed=2**31 - 1, noise_anim=True)
    # 2**34 is a whole number of 32 bit wraps, so the seed comes back untouched.
    assert effective_noise_seed(params, 2**34) == 2**31 - 1
    # One step past the sign bit, which a 31 bit mask would fold back to zero.
    assert effective_noise_seed(params, 1) == 2**31
    assert effective_noise_seed(params, 2**32 + 1) == 2**31


def test_render_frame_passes_noise_mode_to_synthesis():
    seen = []

    def synthesis(ws, noise_mode):
        import torch

        seen.append(noise_mode)
        return torch.zeros([1, 3, 8, 8])

    model = _fake_model(synthesis)
    model.render_frame(render_params(noise_seed=5), 0)
    model.render_frame(render_params(noise_enabled=False), 1)
    assert seen == ["random", "none"]


def test_render_frame_passes_force_fp32_only_when_it_is_set():
    """force_fp32 is legacy, CUDA-only semantics: a no-op wherever the real
    network ignores it, so it is only ever passed through when True, and a
    fake synthesis taking the plain (ws, noise_mode) shape everywhere else in
    this file keeps working with the False (default) path."""
    seen = []

    def synthesis(ws, noise_mode, force_fp32=False):
        import torch

        seen.append(force_fp32)
        return torch.zeros([1, 3, 8, 8])

    model = _fake_model(synthesis)
    model.render_frame(render_params(force_fp32=False), 0)
    model.render_frame(render_params(force_fp32=True), 1)
    assert seen == [False, True]


def test_render_frame_seeds_torch_with_effective_seed(monkeypatch):
    import torch

    seeds = []
    monkeypatch.setattr(torch, "manual_seed", seeds.append)
    model = _fake_model(_zeros_synthesis)
    model.render_frame(render_params(noise_seed=11, noise_anim=True), 4)
    assert seeds == [15]


@pytest.mark.parametrize(
    "changes",
    [
        {"noise_enabled": False},
        {"noise_seed": 0, "noise_anim": False},
    ],
)
def test_render_frame_leaves_the_global_rng_alone_outside_random_noise(
    monkeypatch, changes
):
    # The network never samples in "none" or "const" mode, so reseeding there
    # only pins torch's global stream for whatever else runs on this thread.
    import torch

    seeds = []
    monkeypatch.setattr(torch, "manual_seed", seeds.append)
    model = _fake_model(_zeros_synthesis)
    model.render_frame(render_params(**changes), 3)
    assert seeds == []


# --- vec and loop modes -----------------------------------------------------
#
# These fixtures use z_dim == w_dim (unlike the fixtures above, which keep
# them different to prove shape routing does not care), so the vec/W tests
# below can exercise realistic single-vector-broadcast behavior.


class _RecordingMapping:
    def __init__(self, w_dim, num_ws):
        import torch

        self.w_avg = torch.zeros([w_dim])
        self.c_dim = 0
        self._num_ws = num_ws
        self.calls = []

    def __call__(self, z, c, truncation_psi):
        self.calls.append((z.clone(), truncation_psi))
        return z.unsqueeze(1).repeat(1, self._num_ws, 1) * truncation_psi


class _VecG:
    def __init__(self, z_dim, num_ws, synthesis):
        self.z_dim = z_dim
        self.num_ws = num_ws
        self.mapping = _RecordingMapping(z_dim, num_ws)
        self.synthesis = synthesis


def _recording_synthesis(sink):
    def synthesis(ws, noise_mode):
        import torch

        sink.append(ws.clone())
        return torch.zeros([1, 3, 4, 4])

    return synthesis


def _vec_model(z_dim=4, num_ws=3, synthesis=None):
    import torch

    from autolume.live.core.generator import LoadedModel

    return LoadedModel(
        "/tmp/vec.pkl", _VecG(z_dim, num_ws, synthesis or _zeros_synthesis), torch.device("cpu")
    )


def test_vec_projected_passes_truncation_into_mapping():
    model = _vec_model(z_dim=4, num_ws=3)
    params = render_params(
        vector_mode=True,
        latent_project=True,
        latent_vec=(1.0, 2.0, 3.0, 4.0),
        truncation_psi=0.5,
    )
    model.render_frame(params, 0)
    assert len(model.G.mapping.calls) == 1
    z, psi = model.G.mapping.calls[0]
    assert psi == 0.5
    assert z.tolist() == [[1.0, 2.0, 3.0, 4.0]]


def test_vec_unprojected_exact_length_passes_rows_through():
    sink = []
    model = _vec_model(z_dim=4, num_ws=3, synthesis=_recording_synthesis(sink))
    rows = (1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3)
    params = render_params(
        vector_mode=True, latent_project=False, latent_vec=rows
    )
    model.render_frame(params, 0)
    assert sink[-1][0].tolist() == [
        [1, 1, 1, 1],
        [2, 2, 2, 2],
        [3, 3, 3, 3],
    ]


def test_vec_unprojected_too_many_rows_truncates_to_num_ws():
    sink = []
    model = _vec_model(z_dim=4, num_ws=3, synthesis=_recording_synthesis(sink))
    rows = (1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4)
    params = render_params(
        vector_mode=True, latent_project=False, latent_vec=rows
    )
    model.render_frame(params, 0)
    assert sink[-1][0].tolist() == [
        [1, 1, 1, 1],
        [2, 2, 2, 2],
        [3, 3, 3, 3],
    ]


def test_vec_unprojected_too_few_rows_repeats_last_row():
    sink = []
    model = _vec_model(z_dim=4, num_ws=3, synthesis=_recording_synthesis(sink))
    rows = (1, 1, 1, 1, 2, 2, 2, 2)
    params = render_params(
        vector_mode=True, latent_project=False, latent_vec=rows
    )
    model.render_frame(params, 0)
    assert sink[-1][0].tolist() == [
        [1, 1, 1, 1],
        [2, 2, 2, 2],
        [2, 2, 2, 2],
    ]


def test_empty_vector_fallback_is_deterministic_and_logged_once(caplog):
    import numpy as np
    import torch

    model = _vec_model(z_dim=4, num_ws=2)
    params = render_params(vector_mode=True, latent_project=True, latent_vec=())
    with caplog.at_level(logging.WARNING):
        model.render_frame(params, 0)
        model.render_frame(params, 1)
    calls = model.G.mapping.calls
    assert len(calls) == 2
    expected = np.random.RandomState(0).randn(4).astype(np.float32)
    assert torch.allclose(calls[0][0][0], torch.from_numpy(expected))
    assert torch.allclose(calls[1][0][0], torch.from_numpy(expected))
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1


def test_vec_unprojected_empty_vector_fallback_is_deterministic_and_logged_once(
    caplog,
):
    sink = []
    model = _vec_model(z_dim=4, num_ws=3, synthesis=_recording_synthesis(sink))
    params = render_params(vector_mode=True, latent_project=False, latent_vec=())
    with caplog.at_level(logging.WARNING):
        model.render_frame(params, 0)
        model.render_frame(params, 1)
    first, second = sink[-2].tolist(), sink[-1].tolist()
    assert first == second
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1


def test_loop_endpoint_cache_reuses_mapping_per_endpoint():
    model = _vec_model(z_dim=4, num_ws=3)
    kf0 = Keyframe("vec", vec=(1.0, 2.0, 3.0, 4.0), project=True)
    kf1 = Keyframe("vec", vec=(5.0, 6.0, 7.0, 8.0), project=True)
    params = render_params(
        loop_active=True,
        keyframes=(kf0, kf1),
        loop_index=1,
        loop_alpha=0.5,
        truncation_psi=0.6,
    )
    model.render_frame(params, 0)
    model.render_frame(params, 1)
    assert len(model.G.mapping.calls) == 2


def test_loop_endpoint_cache_is_bounded_to_the_cap():
    model = _vec_model(z_dim=4, num_ws=3)
    keyframes = tuple(
        Keyframe("vec", vec=(float(i),) * 4, project=True) for i in range(6)
    )
    for index in range(len(keyframes)):
        params = render_params(
            loop_active=True,
            keyframes=keyframes,
            loop_index=index,
            loop_alpha=0.5,
            truncation_psi=0.6,
        )
        model.render_frame(params, index)
    assert len(model._keyframe_w_cache) <= 4


def test_loop_vec_keyframe_wrong_length_falls_back_without_raising(caplog):
    model = _vec_model(z_dim=4, num_ws=3)
    kf0 = Keyframe("seed", seed_x=0.0, seed_y=0.0)
    kf1 = Keyframe("vec", vec=(1.0, 2.0), project=True)
    params = render_params(
        loop_active=True,
        keyframes=(kf0, kf1),
        loop_index=1,
        loop_alpha=0.5,
        truncation_psi=0.6,
    )
    with caplog.at_level(logging.WARNING):
        model.render_frame(params, 0)
    assert len(model.G.mapping.calls) == 2
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1


def test_loop_alpha_zero_matches_previous_keyframe():
    import torch

    sink = []
    model = _vec_model(z_dim=4, num_ws=3, synthesis=_recording_synthesis(sink))
    kf0 = Keyframe("seed", seed_x=0.0, seed_y=0.0)
    kf1 = Keyframe("seed", seed_x=1.0, seed_y=0.0)
    loop_params = render_params(
        loop_active=True,
        keyframes=(kf0, kf1),
        loop_index=1,
        loop_alpha=0.0,
        truncation_psi=0.7,
    )
    model.render_frame(loop_params, 0)
    loop_ws = sink[-1]

    seed_params = render_params(latent_x=0.0, latent_y=0.0, truncation_psi=0.7)
    model.render_frame(seed_params, 0)
    seed_ws = sink[-1]

    assert torch.allclose(loop_ws, seed_ws, atol=1e-5)


def test_loop_alpha_one_matches_current_keyframe():
    import torch

    sink = []
    model = _vec_model(z_dim=4, num_ws=3, synthesis=_recording_synthesis(sink))
    kf0 = Keyframe("seed", seed_x=0.0, seed_y=0.0)
    kf1 = Keyframe("seed", seed_x=1.0, seed_y=0.0)
    loop_params = render_params(
        loop_active=True,
        keyframes=(kf0, kf1),
        loop_index=1,
        loop_alpha=1.0,
        truncation_psi=0.7,
    )
    model.render_frame(loop_params, 0)
    loop_ws = sink[-1]

    seed_params = render_params(latent_x=1.0, latent_y=0.0, truncation_psi=0.7)
    model.render_frame(seed_params, 0)
    seed_ws = sink[-1]

    assert torch.allclose(loop_ws, seed_ws, atol=1e-5)


def test_loop_index_zero_wraps_to_last_keyframe():
    import torch

    sink = []
    model = _vec_model(z_dim=4, num_ws=3, synthesis=_recording_synthesis(sink))
    kf0 = Keyframe("seed", seed_x=0.0, seed_y=0.0)
    kf1 = Keyframe("seed", seed_x=1.0, seed_y=0.0)
    kf2 = Keyframe("seed", seed_x=2.0, seed_y=0.0)
    loop_params = render_params(
        loop_active=True,
        keyframes=(kf0, kf1, kf2),
        loop_index=0,
        loop_alpha=0.0,
        truncation_psi=0.7,
    )
    model.render_frame(loop_params, 0)
    loop_ws = sink[-1]

    seed_params = render_params(latent_x=2.0, latent_y=0.0, truncation_psi=0.7)
    model.render_frame(seed_params, 0)
    seed_ws = sink[-1]

    assert torch.allclose(loop_ws, seed_ws, atol=1e-5)


def test_slerp_of_orthogonal_unit_vectors_has_unit_norm():
    import torch

    w0 = torch.tensor([1.0, 0.0])
    w1 = torch.tensor([0.0, 1.0])
    result = slerp(0.5, w0, w1)
    assert abs(float(result.norm()) - 1.0) < 1e-5


def test_slerp_falls_back_to_lerp_when_near_colinear():
    import torch

    w0 = torch.tensor([1.0, 0.0])
    w1 = torch.tensor([1.0, 1e-5])
    alpha = 0.3
    result = slerp(alpha, w0, w1)
    expected = w0 + alpha * (w1 - w0)
    assert torch.allclose(result, expected, atol=1e-6)


def test_slerp_alpha_bounds_reproduce_endpoints():
    import torch

    w0 = torch.tensor([1.0, 0.0, 0.0])
    w1 = torch.tensor([0.0, 1.0, 0.0])
    assert torch.allclose(slerp(0.0, w0, w1), w0, atol=1e-6)
    assert torch.allclose(slerp(1.0, w0, w1), w1, atol=1e-6)


# --- layer catalog enumeration ------------------------------------------


def _block(channels, height, width):
    import torch
    import torch.nn as nn

    class _Block(nn.Module):
        def forward(self, ws):
            return torch.zeros(ws.shape[0], channels, height, width)

    return _Block()


def _fake_synthesis_module():
    import torch.nn as nn

    class _Synthesis(nn.Module):
        def __init__(self):
            super().__init__()
            # conv1 is deliberately non-square: a width/height transposition
            # bug would still pass a square-only fixture.
            self.conv1 = _block(channels=8, height=4, width=6)
            self.torgb = _block(channels=3, height=8, width=8)

        def forward(self, ws, noise_mode="const"):
            self.conv1(ws)
            return self.torgb(ws)

    return _Synthesis()


def _failing_synthesis_module():
    import torch.nn as nn

    class _Boom(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = _block(channels=8, height=4, width=6)

        def forward(self, ws, noise_mode="const"):
            raise RuntimeError("exotic architecture")

    return _Boom()


def _group_cnn_synthesis_module():
    """A single child whose output is 5D: `(N, C, G, H, W)`, the group-CNN
    layout the 4D/5D filter is meant to also accept."""
    import torch
    import torch.nn as nn

    class _GroupBlock(nn.Module):
        def forward(self, ws):
            return torch.zeros(ws.shape[0], 6, 3, 4, 5)

    class _Synthesis(nn.Module):
        def __init__(self):
            super().__init__()
            self.group_conv = _GroupBlock()

        def forward(self, ws, noise_mode="const"):
            return self.group_conv(ws)

    return _Synthesis()


def test_enumerate_layers_records_names_shapes_and_order():
    model = _fake_model(_fake_synthesis_module())

    layers = model.enumerate_layers()

    assert layers == (
        LayerInfo(name="conv1", channels=8, width=6, height=4),
        LayerInfo(name="torgb", channels=3, width=8, height=8),
        LayerInfo(name="output", channels=3, width=8, height=8),
    )


def test_enumerate_layers_5d_output_channels_read_axis_1_not_axis_minus_3():
    # 5D layout is (N, C, G, H, W): axis -3 is the group count (3), not the
    # channel count (6). channels must come from axis 1.
    model = _fake_model(_group_cnn_synthesis_module())

    layers = model.enumerate_layers()

    assert layers == (
        LayerInfo(name="group_conv", channels=6, width=5, height=4),
        LayerInfo(name="output", channels=6, width=5, height=4),
    )


def test_enumerate_layers_removes_hooks_afterward():
    import torch

    synthesis = _fake_synthesis_module()
    model = _fake_model(synthesis)

    model.enumerate_layers()

    for module in synthesis.modules():
        assert len(module._forward_hooks) == 0

    # A second call to the same (now unhooked) synthesis must not raise.
    synthesis(torch.zeros(1, 2, 8))


def test_enumerate_layers_failure_yields_empty_catalog_and_logs_once(caplog):
    synthesis = _failing_synthesis_module()
    model = _fake_model(synthesis)

    with caplog.at_level(logging.WARNING):
        layers = model.enumerate_layers()

    assert layers == ()
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    # The safety-critical half of the finally contract: a raising dry pass
    # must not leave hooks on a model about to render.
    for module in synthesis.modules():
        assert len(module._forward_hooks) == 0


def test_model_host_publishes_empty_layer_catalog_when_enumeration_fails():
    import torch

    from autolume.live.core.generator import LoadedModel

    def loader(path):
        return LoadedModel(
            path, _FakeG(_failing_synthesis_module()), torch.device("cpu")
        )

    host = ModelHost(loader=loader)
    host.request_load("/tmp/exotic.pkl")
    deadline = time.monotonic() + 2.0
    while host.info_store.snapshot() is None and time.monotonic() < deadline:
        time.sleep(0.005)

    info = host.info_store.snapshot()
    assert info is not None
    assert info.layers == ()
    assert host.current() is not None
    assert host.error() is None
    host.stop()


def test_model_host_publishes_empty_layer_catalog_when_wrapper_enumeration_raises(
    caplog,
):
    """`_model_info`'s own guard, not `LoadedModel.enumerate_layers`'s.

    A future wrapper generator (a mixed-model network, say) need not be a
    `LoadedModel`: it only has to duck-type `z_dim`/`num_ws`/`enumerate_layers`.
    If its `enumerate_layers` raises, `_model_info` must still publish an
    empty catalog rather than failing the whole load.
    """

    class _WrapperModel:
        def __init__(self, path):
            self.pkl_path = path
            self.z_dim = 4
            self.num_ws = 2

        def enumerate_layers(self):
            raise RuntimeError("wrapper enumeration exploded")

    host = ModelHost(loader=_WrapperModel)
    with caplog.at_level(logging.WARNING):
        host.request_load("/tmp/wrapper.pkl")
        deadline = time.monotonic() + 2.0
        while host.info_store.snapshot() is None and time.monotonic() < deadline:
            time.sleep(0.005)

    info = host.info_store.snapshot()
    assert info is not None
    assert info.layers == ()
    assert host.current() is not None
    assert host.error() is None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert warnings[0].exc_info is not None
    host.stop()


def test_model_host_publishes_layer_catalog_on_successful_load():
    import torch

    from autolume.live.core.generator import LoadedModel

    def loader(path):
        return LoadedModel(
            path, _FakeG(_fake_synthesis_module()), torch.device("cpu")
        )

    host = ModelHost(loader=loader)
    host.request_load("/tmp/good.pkl")
    deadline = time.monotonic() + 2.0
    while host.info_store.snapshot() is None and time.monotonic() < deadline:
        time.sleep(0.005)

    info = host.info_store.snapshot()
    assert info is not None
    assert [layer.name for layer in info.layers] == ["conv1", "torgb", "output"]
    host.stop()


# --- image derivation ----------------------------------------------------


def _activation(channels, height=1, width=2):
    import torch

    count = channels * height * width
    return torch.arange(count, dtype=torch.float32).reshape(channels, height, width)


def test_channel_window_takes_three_consecutive_channels():
    window = channel_window(_activation(5), base_channel=1, grayscale=False)
    assert window.tolist() == [[[2.0, 3.0]], [[4.0, 5.0]], [[6.0, 7.0]]]


def test_channel_window_clamps_the_base_to_keep_three_channels_in_range():
    window = channel_window(_activation(5), base_channel=9, grayscale=False)
    assert window.tolist() == [[[4.0, 5.0]], [[6.0, 7.0]], [[8.0, 9.0]]]


def test_channel_window_grayscale_replicates_one_channel():
    window = channel_window(_activation(5), base_channel=3, grayscale=True)
    assert window.tolist() == [[[6.0, 7.0]], [[6.0, 7.0]], [[6.0, 7.0]]]


def test_channel_window_grayscale_clamps_the_base_to_the_last_channel():
    window = channel_window(_activation(5), base_channel=99, grayscale=True)
    assert window.tolist() == [[[8.0, 9.0]], [[8.0, 9.0]], [[8.0, 9.0]]]


@pytest.mark.parametrize("channels", [1, 2])
def test_channel_window_falls_back_to_one_channel_when_three_do_not_exist(channels):
    window = channel_window(_activation(channels), base_channel=1, grayscale=False)
    last = _activation(channels)[channels - 1].tolist()
    assert window.tolist() == [last, last, last]


def _signed_activation():
    import torch

    return torch.tensor([[[1.0, -2.0]], [[0.0, 0.0]], [[3.0, 4.0]]])


def test_derive_normalizes_each_channel_by_its_own_max_absolute_value():
    image = derive_float_image(_signed_activation(), render_params(img_normalize=True))
    assert image.tolist() == [[[0.5, -1.0]], [[0.0, 0.0]], [[0.75, 1.0]]]


def test_derive_scales_by_decibels():
    image = derive_float_image(_signed_activation(), render_params(img_scale_db=20.0))
    assert image.tolist() == [[[10.0, -20.0]], [[0.0, 0.0]], [[30.0, 40.0]]]


def test_derive_normalizes_before_scaling():
    # Normalization is scale invariant, so the other order would swallow the
    # decibel gain entirely and leave the normalized values behind.
    image = derive_float_image(
        _signed_activation(), render_params(img_normalize=True, img_scale_db=20.0)
    )
    assert image.tolist() == [[[5.0, -10.0]], [[0.0, 0.0]], [[7.5, 10.0]]]


def test_derive_keeps_an_all_zero_channel_at_zero_instead_of_nan():
    import torch

    image = derive_float_image(torch.zeros([3, 1, 2]), render_params(img_normalize=True))
    assert image.tolist() == [[[0.0, 0.0]], [[0.0, 0.0]], [[0.0, 0.0]]]
    assert bool(torch.isfinite(image).all())


def test_uint8_frame_maps_the_signed_unit_range_onto_the_byte_range():
    import torch

    image = torch.tensor([[[-1.0, -0.5, 0.0, 0.5, 1.0]]]).repeat(3, 1, 1)
    frame = to_uint8_frame(image)
    assert frame.shape == (1, 5, 3)
    assert frame.dtype.name == "uint8"
    assert frame[0, :, 0].tolist() == [0, 64, 128, 191, 255]


def test_uint8_frame_clamps_out_of_range_values():
    import torch

    image = torch.tensor([[[-4.0, 4.0]]]).repeat(3, 1, 1)
    frame = to_uint8_frame(image)
    assert frame[0, :, 0].tolist() == [0, 255]


def test_uint8_frame_is_contiguous_after_the_channel_permute():
    import torch

    frame = to_uint8_frame(torch.zeros([3, 2, 4]))
    assert frame.flags["C_CONTIGUOUS"]


def _channel_ramp_synthesis(channels=5):
    def synthesis(ws, noise_mode):
        import torch

        ramp = torch.arange(channels, dtype=torch.float32) / channels
        return ramp.reshape(1, channels, 1, 1).repeat(1, 1, 1, 2)

    return synthesis


def test_render_frame_derives_from_the_selected_channels():
    model = _fake_model(_channel_ramp_synthesis())
    frame = model.render_frame(render_params(base_channel=1), 0)
    # Channels 1, 2 and 3 of the ramp: 0.2, 0.4 and 0.6.
    assert frame[0, 0].tolist() == [153, 179, 204]


def test_render_frame_grayscale_replicates_one_channel():
    model = _fake_model(_channel_ramp_synthesis())
    frame = model.render_frame(render_params(base_channel=1, grayscale=True), 0)
    assert frame.shape == (1, 2, 3)
    assert frame[0, 0].tolist() == [153, 153, 153]


# --- super-res wiring ------------------------------------------------------


class _RecordingSuperRes:
    """Stands in for the real stage: records the tensor and device it is
    handed and returns a recognizably different image, so the wiring can be
    checked by what actually came out, not just that a frame came back."""

    def __init__(self):
        self.calls = []

    def apply(self, image, device):
        self.calls.append((image, device))
        return image + 0.5


def test_render_frame_never_touches_super_res_when_disabled():
    model = _fake_model(_channel_ramp_synthesis())
    stage = _RecordingSuperRes()
    model._superres = stage
    frame = model.render_frame(render_params(base_channel=1, use_superres=False), 0)
    assert stage.calls == []
    assert frame[0, 0].tolist() == [153, 179, 204]


def test_render_frame_passes_the_float_image_through_super_res_before_quantizing():
    model = _fake_model(_channel_ramp_synthesis())
    stage = _RecordingSuperRes()
    model._superres = stage
    frame = model.render_frame(render_params(base_channel=1, use_superres=True), 0)
    assert len(stage.calls) == 1
    seen_image, _ = stage.calls[0]
    assert seen_image.dtype.is_floating_point
    assert seen_image.shape == (3, 1, 2)
    # Channels 1-3 of the ramp are 0.2, 0.4, 0.6; the stage's own +0.5 lands
    # on 0.7, 0.9, 1.1 (clamped), which only shows up in the frame if the
    # stage ran before the uint8 conversion, not after it.
    assert frame[0, 0].tolist() == [217, 242, 255]


def test_render_frame_hands_super_res_the_models_device():
    import torch

    model = _fake_model(_channel_ramp_synthesis())
    stage = _RecordingSuperRes()
    model._superres = stage
    model.render_frame(render_params(use_superres=True), 0)
    _, seen_device = stage.calls[0]
    assert seen_device is model.device
    assert seen_device == torch.device("cpu")


def test_render_frame_still_produces_a_frame_when_super_res_returns_unchanged():
    """SuperRes itself guarantees it never raises, returning the original
    image unchanged on any internal failure. The wiring must not assume
    upscaling actually happened, or wrap `apply()` in its own exception
    handling that could mask a genuine bug in the stage: it just uses
    whatever `apply()` hands back."""

    class _UnchangedSuperRes:
        def apply(self, image, device):
            return image

    model = _fake_model(_channel_ramp_synthesis())
    model._superres = _UnchangedSuperRes()
    frame = model.render_frame(render_params(base_channel=1, use_superres=True), 0)
    assert frame.shape == (1, 2, 3)
    assert frame[0, 0].tolist() == [153, 179, 204]


# --- adjuster direction --------------------------------------------------


def test_adjust_weights_are_returned_in_slot_order():
    params = render_params(adjust_w1=1.0, adjust_w5=-2.0, adjust_w8=3.0)
    assert adjust_weights(params) == (1.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 3.0)


def test_direction_delta_is_none_when_every_weight_is_zero():
    params = render_params(directions=((1.0, 2.0, 3.0, 4.0),))
    assert direction_delta(params, 4) == (None, ())


def test_direction_delta_is_none_when_no_direction_is_loaded():
    params = render_params(adjust_w1=3.0)
    assert direction_delta(params, 4) == (None, ())


def test_direction_delta_scales_one_direction_by_its_weight():
    params = render_params(directions=((1.0, 0.0, -0.5, 0.0),), adjust_w1=2.0)
    delta, mismatched = direction_delta(params, 4)
    assert delta.tolist() == [2.0, 0.0, -1.0, 0.0]
    assert mismatched == ()


def test_direction_delta_sums_every_weighted_direction():
    params = render_params(
        directions=((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0)),
        adjust_w1=2.0,
        adjust_w2=0.0,
        adjust_w3=-1.0,
    )
    delta, mismatched = direction_delta(params, 4)
    assert delta.tolist() == [2.0, 0.0, -1.0, 0.0]
    assert mismatched == ()


def test_direction_delta_ignores_weights_past_the_loaded_directions():
    params = render_params(directions=((1.0, 1.0, 1.0, 1.0),), adjust_w1=1.0, adjust_w4=9.0)
    delta, mismatched = direction_delta(params, 4)
    assert delta.tolist() == [1.0, 1.0, 1.0, 1.0]
    assert mismatched == ()


def test_direction_delta_reports_a_wrong_width_instead_of_contributing_it():
    params = render_params(
        directions=((1.0, 2.0), (0.0, 0.0, 1.0, 0.0)), adjust_w1=5.0, adjust_w2=2.0
    )
    delta, mismatched = direction_delta(params, 4)
    assert delta.tolist() == [0.0, 0.0, 2.0, 0.0]
    assert mismatched == (0,)


def test_direction_delta_stays_quiet_about_a_wrong_width_nobody_asked_for():
    params = render_params(directions=((1.0, 2.0),), adjust_w1=0.0)
    assert direction_delta(params, 4) == (None, ())


def test_render_frame_adds_the_weighted_direction_to_w():
    sink = []
    model = _vec_model(z_dim=4, num_ws=3, synthesis=_recording_synthesis(sink))
    model.render_frame(render_params(), 0)
    plain = sink[-1]
    model.render_frame(
        render_params(directions=((1.0, 0.0, -0.5, 0.0),), adjust_w1=2.0), 1
    )
    shifted = sink[-1]
    assert (shifted - plain)[0].tolist() == [[2.0, 0.0, -1.0, 0.0]] * 3


def test_render_frame_leaves_w_alone_when_no_weight_is_set():
    import torch

    sink = []
    model = _vec_model(z_dim=4, num_ws=3, synthesis=_recording_synthesis(sink))
    model.render_frame(render_params(), 0)
    plain = sink[-1]
    model.render_frame(render_params(directions=((1.0, 0.0, -0.5, 0.0),)), 1)
    assert torch.equal(sink[-1], plain)


def test_render_frame_logs_a_wrong_width_direction_once(caplog):
    import torch

    sink = []
    model = _vec_model(z_dim=4, num_ws=3, synthesis=_recording_synthesis(sink))
    model.render_frame(render_params(), 0)
    plain = sink[-1]
    params = render_params(directions=((1.0, 2.0),), adjust_w1=1.0)
    with caplog.at_level(logging.WARNING):
        model.render_frame(params, 1)
        model.render_frame(params, 2)
    assert torch.equal(sink[-1], plain)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1


# --- per layer module attributes -----------------------------------------


def _state_block(**attrs):
    import torch.nn as nn

    class _StateBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.writes = []
            for name, value in attrs.items():
                object.__setattr__(self, name, value)

        def __setattr__(self, name, value):
            if name in ("global_noise", "noise_regulator", "ratio"):
                self.writes.append((name, value))
            super().__setattr__(name, value)

    return _StateBlock()


def _stateful_synthesis(**blocks):
    import torch
    import torch.nn as nn

    class _Synthesis(nn.Module):
        def __init__(self):
            super().__init__()
            for name, block in blocks.items():
                setattr(self, name, block)

        def forward(self, ws, noise_mode="const"):
            return torch.zeros([1, 3, 2, 2])

    return _Synthesis()


def test_module_state_is_written_only_where_the_attribute_already_exists():
    noisy = _state_block(global_noise=1.0)
    plain = _state_block()
    model = _fake_model(_stateful_synthesis(conv1=noisy, torgb=plain))

    model.render_frame(render_params(global_noise=0.25), 0)

    assert noisy.writes == [("global_noise", 0.25)]
    assert plain.writes == []
    assert not hasattr(plain, "global_noise")


def test_module_state_is_not_rewritten_while_nothing_moves():
    noisy = _state_block(global_noise=1.0)
    model = _fake_model(_stateful_synthesis(conv1=noisy))

    model.render_frame(render_params(global_noise=0.25), 0)
    model.render_frame(render_params(global_noise=0.25), 1)
    assert noisy.writes == [("global_noise", 0.25)]

    model.render_frame(render_params(global_noise=0.75), 2)
    assert noisy.writes == [("global_noise", 0.25), ("global_noise", 0.75)]


def test_module_state_is_not_rewritten_while_a_layer_override_holds():
    # Each snapshot carries its own dict, so holding an override constant is
    # what exercises the change cache's dict comparison rather than two
    # empty dicts comparing equal.
    conv1 = _state_block(global_noise=1.0, noise_regulator=0)
    model = _fake_model(_stateful_synthesis(conv1=conv1))

    model.render_frame(render_params(layer_noise=(("conv1", 0.7),)), 0)
    after_first = list(conv1.writes)
    model.render_frame(render_params(layer_noise=(("conv1", 0.7),)), 1)
    assert conv1.writes == after_first

    model.render_frame(render_params(layer_noise=(("conv1", 0.9),)), 2)
    assert conv1.writes == after_first + [
        ("global_noise", 1.0),
        ("noise_regulator", 0.9),
    ]


def test_per_layer_noise_strength_reaches_its_layer_and_nothing_else():
    conv1 = _state_block(global_noise=1.0, noise_regulator=0)
    torgb = _state_block(global_noise=1.0, noise_regulator=0)
    model = _fake_model(_stateful_synthesis(conv1=conv1, torgb=torgb))

    model.render_frame(render_params(layer_noise=(("conv1", 0.7),)), 0)

    assert ("noise_regulator", 0.7) in conv1.writes
    assert ("noise_regulator", 0.0) in torgb.writes


def test_a_removed_noise_strength_returns_the_layer_to_neutral():
    conv1 = _state_block(noise_regulator=0)
    model = _fake_model(_stateful_synthesis(conv1=conv1))

    model.render_frame(render_params(layer_noise=(("conv1", 0.7),)), 0)
    model.render_frame(render_params(), 1)

    assert conv1.writes == [("noise_regulator", 0.7), ("noise_regulator", 0.0)]


def test_per_layer_ratio_reaches_its_layer_and_the_rest_stay_square():
    """Note the swap: `(rx, ry)` in state arrives as `(ry, rx)` on the module.

    `SynthesisLayer.forward` reads slot 0 as the height scale, so the push is
    where the panel's x-then-y order is turned into the layer's own. Pinned
    here as well as in the test below, because this is the assertion that
    would go quietly wrong if the swap were ever moved outward into state.
    """
    conv1 = _state_block(ratio=(1, 1))
    torgb = _state_block(ratio=(1, 1))
    model = _fake_model(_stateful_synthesis(conv1=conv1, torgb=torgb))

    model.render_frame(render_params(layer_ratios=(("conv1", 2.0, 3.0),)), 0)

    assert conv1.writes == [("ratio", (3.0, 2.0))]
    assert torgb.writes == [("ratio", (1.0, 1.0))]


def test_the_ratio_push_swaps_the_pair_and_state_keeps_ui_order():
    """"Ratio x" has to scale width, which means swapping at the push.

    The layer resizes to `(in_w * rx, in_h * ry)` with `in_w` bound to
    `x.shape[-2]`, the height, so slot 0 of the module's own pair is the
    height scale. Measured on a real model: a "Ratio x" of 2 on a 1024 model
    rendered a 2048 by 1024 frame before this swap, which is the y axis.

    The other half of the fix is that nothing outside the push moved, so no
    preset needs migrating: state and `RenderParams` are asserted to still
    hold the pair the panel wrote, in the panel's order.
    """
    conv1 = _state_block(ratio=(1, 1))
    model = _fake_model(_stateful_synthesis(conv1=conv1))
    state = ControlState(layer_ratios=(("conv1", 2.0, 1.0),))
    params = to_render_params(state)

    model.render_frame(params, 0)

    assert conv1.ratio == (1.0, 2.0)
    assert state.layer_ratios == (("conv1", 2.0, 1.0),)
    assert params.layer_ratios == {"conv1": (2.0, 1.0)}


def test_a_removed_ratio_returns_the_layer_to_neutral_either_way_round():
    """Neutral is its own mirror, so the swap cannot leave a layer stretched."""
    conv1 = _state_block(ratio=(1, 1))
    model = _fake_model(_stateful_synthesis(conv1=conv1))

    model.render_frame(render_params(layer_ratios=(("conv1", 2.0, 0.5),)), 0)
    model.render_frame(render_params(), 1)

    assert conv1.writes == [("ratio", (0.5, 2.0)), ("ratio", (1.0, 1.0))]


def test_the_ratio_swap_does_not_leak_past_the_push():
    """Written through OSC, saved, reloaded, pushed: x stays first everywhere.

    The pair is swapped at the module write and nowhere else, so no preset
    needs migrating and an OSC sender never has to know about any of it.
    """
    state = apply_event(
        ControlState(), ControlEvent(BEND_RATIO, SetLayerRatio("conv1", 2.0, 0.5))
    )
    assert state.layer_ratios == (("conv1", 2.0, 0.5),)

    payload = presets.to_payload(state)
    assert payload["layer_ratios"] == [{"layer": "conv1", "rx": 2.0, "ry": 0.5}]
    reloaded = presets.from_payload(payload)
    assert reloaded.layer_ratios == (("conv1", 2.0, 0.5),)

    conv1 = _state_block(ratio=(1, 1))
    model = _fake_model(_stateful_synthesis(conv1=conv1))
    model.render_frame(render_params(layer_ratios=reloaded.layer_ratios), 0)
    assert conv1.ratio == (0.5, 2.0)


def test_the_ratio_push_still_only_walks_the_network_when_something_moved():
    """The swap must not cost a walk per frame.

    `_apply_module_state` is memoized on the snapshot's own values, and the
    swap happens after that compare rather than by rewriting the mapping it
    keys on, so a held ratio is still one walk.
    """
    conv1 = _state_block(ratio=(1, 1))
    model = _fake_model(_stateful_synthesis(conv1=conv1))

    model.render_frame(render_params(layer_ratios=(("conv1", 2.0, 0.5),)), 0)
    after_first = list(conv1.writes)
    model.render_frame(render_params(layer_ratios=(("conv1", 2.0, 0.5),)), 1)

    assert conv1.writes == after_first


# --- ratios against the real synthesis layer -----------------------------
#
# Everything above pushes onto a fake module. These run the real
# `custom_stylegan2.SynthesisLayer`, which is what the loader rebuilds every
# stylegan2 pkl into (`legacy.create_networks`), so they are the tests that
# would have caught the freeze: the layer's random noise branch drew its field
# at the layer's nominal size while the activation beside it had been resized
# by the ratio, and `modulated_conv2d` raised on every frame.


def _ratio_model(img_resolution=32):
    """A small real generator, on the CPU, wrapped as a `LoadedModel`.

    Real rather than a double because the defect was in the synthesis layer
    itself. Small enough that a handful of frames costs milliseconds.
    """
    import torch

    from architectures import custom_stylegan2

    torch.manual_seed(0)
    G = (
        custom_stylegan2.Generator(
            z_dim=8,
            c_dim=0,
            w_dim=8,
            img_channels=3,
            img_resolution=img_resolution,
            synthesis_kwargs={"channel_base": 64, "channel_max": 16},
        )
        .eval()
        .requires_grad_(False)
    )
    return LoadedModel("/tmp/ratio.pkl", G, torch.device("cpu"))


def _noisy_ratio_params(**changes):
    """Params with a ratio on one layer and that layer's noise turned up.

    A freshly built generator's `noise_strength` is zero, so the noise field
    reaches the picture only through `noise_regulator`, which is what
    `layer_noise` writes. Without it an animated frame is bit identical to a
    still one and the animation assertion below would pass on nothing.
    """
    return render_params(
        layer_ratios=(("b8.conv0", 2.0, 1.0),),
        layer_noise=(("b8.conv0", 1.0), ("b8.conv1", 1.0), ("b16.conv0", 1.0)),
        **changes,
    )


def test_a_ratio_renders_with_animated_noise_and_the_noise_actually_moves():
    """The freeze, against the layer that had it.

    Three consecutive frames, with the seed advancing on each, so this covers
    the frame rendering at all *and* the animation still animating. A fix that
    quietly held the mode on const would pass the first assertion and fail the
    second.
    """
    import numpy as np

    model = _ratio_model()
    params = _noisy_ratio_params(noise_anim=True, noise_seed=7)
    assert noise_mode(params) == "random"

    frames = [model.render_frame(params, index) for index in range(3)]

    # "Ratio x" of 2 on b8.conv0 widens every layer after it, and the whole
    # frame with them.
    assert [frame.shape for frame in frames] == [(32, 64, 3)] * 3
    assert not np.array_equal(frames[0], frames[1])
    assert not np.array_equal(frames[1], frames[2])


def test_a_ratio_gives_the_same_frame_shape_in_every_noise_mode():
    """The random field is sized like the const one, so the shapes agree."""
    model = _ratio_model()
    shapes = {
        mode: model.render_frame(params, 0).shape
        for mode, params in (
            ("random", _noisy_ratio_params(noise_anim=True)),
            ("const", _noisy_ratio_params()),
            ("none", _noisy_ratio_params(noise_enabled=False)),
        )
    }
    assert set(shapes.values()) == {(32, 64, 3)}


def test_animated_noise_is_still_deterministic_under_a_ratio():
    import numpy as np

    model = _ratio_model()
    params = _noisy_ratio_params(noise_anim=True, noise_seed=7)
    assert np.array_equal(
        model.render_frame(params, 2), model.render_frame(params, 2)
    )


def _ratio_probe_params(rx, ry, layer="b16.conv0", **changes):
    """A ratio on one layer, with the noise turned up on the layers around it.

    `b16.conv0` by default because it is the layer that takes an 8 pixel
    activation up by 2, the size where a truncation on the wrong side of the
    upsample first disagrees with the activation.
    """
    return render_params(
        layer_ratios=((layer, rx, ry),),
        layer_noise=(
            ("b8.conv0", 1.0),
            ("b8.conv1", 1.0),
            ("b16.conv0", 1.0),
            ("b16.conv1", 1.0),
        ),
        **changes,
    )


def _observed_layer_sizes(model, params, index=0):
    """What the real layers did, read off the tensors they handed the conv.

    Each row is `(layer, activation, noise, up)` where `activation` is the
    size the layer resized its input to and `noise` is the field it passed
    alongside it, both taken from the tensors themselves rather than
    recomputed from the layer's own formula. That is the whole point of
    reading it this way: a row is evidence about the layer, not a second copy
    of the expression under test, which is what let a wrong expression pass.

    The convolution's own `up` is captured too, because the noise is added
    after the upsample and so has to be `up` times the activation.
    """
    import torch

    from architectures import custom_stylegan2

    rows = []
    entered = []
    real_conv = custom_stylegan2.modulated_conv2d

    def spy(*args, **kwargs):
        noise = kwargs.get("noise")
        # `ToRGBLayer` goes through the same function with no noise at all.
        if noise is not None:
            rows.append(
                (
                    entered[-1],
                    tuple(int(n) for n in kwargs["x"].shape[-2:]),
                    tuple(int(n) for n in noise.shape[-2:]),
                    int(kwargs.get("up", 1)),
                )
            )
        return real_conv(*args, **kwargs)

    handles = [
        module.register_forward_pre_hook(lambda module, inputs: entered.append(module))
        for module in model.G.synthesis.modules()
        if hasattr(module, "ratio") and getattr(module, "use_noise", False)
    ]
    custom_stylegan2.modulated_conv2d = spy
    try:
        with torch.no_grad():
            frame = model.render_frame(params, index)
    finally:
        custom_stylegan2.modulated_conv2d = real_conv
        for handle in handles:
            handle.remove()
    return frame, rows


def test_a_neutral_ratio_leaves_the_random_field_at_its_nominal_size():
    """Why the layer's edit is a no-op for training and for every still look.

    At ratio (1, 1) the field the layer actually draws has to come out at
    exactly `resolution * init_res // 4`, the size it used to be hard coded
    to, for every layer in the ladder: `conv0` layers upsample (`up=2`) and
    `conv1` layers do not, and both are checked here. The generator's own
    constant noise buffer is that nominal size too, so this pins the two
    branches to one grid. Read off the real tensors, so a layer that sized
    its field some other way that happens to agree with the old formula still
    fails here.
    """
    model = _ratio_model()
    frame, rows = _observed_layer_sizes(model, render_params(noise_anim=True))

    assert frame.shape == (32, 32, 3)
    assert len(rows) >= 7
    for layer, activation, noise, up in rows:
        nominal = (
            layer.resolution * layer.init_res[0] // 4,
            layer.resolution * layer.init_res[1] // 4,
        )
        assert noise == (activation[0] * up, activation[1] * up)
        assert noise == nominal
        assert tuple(layer.noise_const.shape) == nominal


# The ratios below are deliberately not 2, 0.5 or 3. On an integral ratio a
# truncation before the upsample and one after it give the same number, which
# is why a whole band of broken sizes went unnoticed: 1.1 on an 8 pixel
# activation with `up=2` draws 17 against an activation of 16, and 0.2 on the
# same layer takes the activation itself to zero pixels.
@pytest.mark.parametrize(
    "rx, ry, frame_shape",
    [
        (1.1, 1.1, (32, 32, 3)),
        (1.1, 1.0, (32, 32, 3)),
        (0.7, 0.7, (20, 20, 3)),
        (0.2, 0.2, (4, 4, 3)),
        (3.3, 0.15, (4, 104, 3)),
    ],
)
@pytest.mark.parametrize("anim", [True, False])
def test_the_noise_field_follows_the_activation_at_awkward_ratios(
    rx, ry, frame_shape, anim
):
    """The field the layer draws is the size the convolution will produce.

    Both noise modes, because the const branch resizes its buffer with the
    same expression and carried the same rounding.
    """
    model = _ratio_model()
    params = _ratio_probe_params(rx, ry, noise_anim=anim)
    frame, rows = _observed_layer_sizes(model, params)

    assert frame.shape == frame_shape
    assert len(rows) >= 7
    for _layer, activation, noise, up in rows:
        assert noise == (activation[0] * up, activation[1] * up)


def test_a_ratio_small_enough_to_flatten_two_layers_keeps_one_grid():
    """The floor is on the activation size, not bolted onto the noise size.

    Small ratios compound: `b8.conv0` takes a 4 pixel activation to zero and
    `b16.conv0` then works on what is left of it. A floor applied to the noise
    on its own survives that too, but by handing the convolution a 1 by 1
    field that broadcasts silently over the whole activation. Asserting the
    noise against the activation rather than against 1 is what tells the two
    apart.
    """
    model = _ratio_model()
    params = render_params(
        layer_ratios=(("b8.conv0", 0.2, 0.2), ("b16.conv0", 0.2, 0.2)),
        layer_noise=(("b8.conv0", 1.0), ("b16.conv0", 1.0)),
        noise_anim=True,
    )
    frame, rows = _observed_layer_sizes(model, params)

    assert frame.shape == (4, 4, 3)
    assert (1, 1) in [activation for _l, activation, _n, _u in rows]
    for _layer, activation, noise, up in rows:
        assert noise == (activation[0] * up, activation[1] * up)


def test_a_hand_edited_negative_ratio_renders_instead_of_raising():
    """Nothing between the preset and the layer clamps this, so the layer must.

    A ratio can only reach the layer through a preset file or an OSC message,
    and neither validates the sign. Before the floor, a negative one asked
    `kornia` for a negative sized activation.
    """
    payload = presets.to_payload(ControlState())
    payload["layer_ratios"] = [{"layer": "b16.conv0", "rx": -1.0, "ry": -0.5}]
    reloaded = presets.from_payload(payload)
    assert reloaded.layer_ratios == (("b16.conv0", -1.0, -0.5),)

    model = _ratio_model()
    frame, rows = _observed_layer_sizes(
        model,
        render_params(
            layer_ratios=reloaded.layer_ratios,
            layer_noise=(("b16.conv0", 1.0), ("b16.conv1", 1.0)),
            noise_anim=True,
        ),
    )

    assert frame.shape[2] == 3
    for _layer, activation, noise, up in rows:
        assert min(activation) >= 1
        assert noise == (activation[0] * up, activation[1] * up)


# --- bending hooks, transforms and layer capture --------------------------
#
# The fixture chains two layers with different gains so a transform applied
# at the wrong point in the network reads back a different number, and uses
# 0.25 rather than 0.5 as its base so that inversion is not its own inverse.


def _bendable_synthesis(channels=3, height=1, width=2, tuple_output=False):
    import torch
    import torch.nn as nn

    class _Hooked(nn.Module):
        def __init__(self, gain):
            super().__init__()
            self.gain = gain
            self.hook_registrations = 0

        def register_forward_hook(self, hook, **kwargs):
            self.hook_registrations += 1
            return super().register_forward_hook(hook, **kwargs)

        def forward(self, x):
            return x * self.gain

    class _Synthesis(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = _Hooked(1.0)
            self.torgb = _Hooked(0.5)

        def forward(self, ws, noise_mode="const"):
            base = torch.full([1, channels, height, width], 0.25)
            image = self.torgb(self.conv1(base))
            return (image, []) if tuple_output else image

    return _Synthesis()


def _bendable_model(**changes):
    return _fake_model(_bendable_synthesis(**changes))


def _pixel(frame):
    return frame[0, 0].tolist()


# conv1 sees 0.25, torgb turns it into 0.125, which quantizes to 143.
_UNBENT = [143, 143, 143]
_ALL_CHANNELS = (0, 1, 2)


def test_no_hooks_are_registered_without_transforms_or_a_capture_layer():
    model = _bendable_model()
    assert _pixel(model.render_frame(render_params(), 0)) == _UNBENT
    for module in model.G.synthesis.modules():
        assert len(module._forward_hooks) == 0
    assert model.G.synthesis.conv1.hook_registrations == 0


def test_one_hook_is_registered_per_bent_layer_and_reused_across_frames():
    model = _bendable_model()
    params = render_params(
        transforms=(Transform("ablate", "conv1", (1.0,), _ALL_CHANNELS),)
    )
    for index in range(4):
        assert _pixel(model.render_frame(params, index)) == [128, 128, 128]
    assert model.G.synthesis.conv1.hook_registrations == 1
    assert model.G.synthesis.torgb.hook_registrations == 0


def test_changing_only_transform_parameters_leaves_the_hooks_in_place():
    model = _bendable_model()
    for index, factor in enumerate((2.0, 3.0, 4.0)):
        model.render_frame(
            render_params(
                transforms=(
                    Transform("scalar-multiply", "conv1", (factor,), _ALL_CHANNELS),
                )
            ),
            index,
        )
    assert model.G.synthesis.conv1.hook_registrations == 1


def test_adding_a_layer_to_the_chain_rebuilds_the_hook_set():
    model = _bendable_model()
    first = Transform("ablate", "conv1", (1.0,), _ALL_CHANNELS)
    second = Transform("ablate", "torgb", (1.0,), _ALL_CHANNELS)
    model.render_frame(render_params(transforms=(first,)), 0)
    model.render_frame(render_params(transforms=(first, second)), 1)
    synthesis = model.G.synthesis
    assert synthesis.torgb.hook_registrations == 1
    # The whole set is rebuilt, so conv1's hook is registered a second time,
    # and re-registered rather than doubled up: one live hook per module.
    assert synthesis.conv1.hook_registrations == 2
    assert len(synthesis.conv1._forward_hooks) == 1
    assert len(synthesis.torgb._forward_hooks) == 1


def test_clearing_the_chain_removes_every_hook():
    model = _bendable_model()
    transform = Transform("ablate", "conv1", (1.0,), _ALL_CHANNELS)
    model.render_frame(render_params(transforms=(transform,)), 0)
    assert _pixel(model.render_frame(render_params(), 1)) == _UNBENT
    for module in model.G.synthesis.modules():
        assert len(module._forward_hooks) == 0


def test_a_capture_layer_alone_registers_its_hook():
    model = _bendable_model()
    model.render_frame(render_params(capture_layer="conv1"), 0)
    assert model.G.synthesis.conv1.hook_registrations == 1
    assert model.G.synthesis.torgb.hook_registrations == 0


@pytest.mark.parametrize(
    "layer,expected",
    [("conv1", [175, 175, 175]), ("torgb", [239, 239, 239])],
)
def test_a_transform_applies_at_the_layer_it_names(layer, expected):
    model = _bendable_model()
    params = render_params(
        transforms=(Transform("invert", layer, (1.0,), _ALL_CHANNELS),)
    )
    assert _pixel(model.render_frame(params, 0)) == expected


@pytest.mark.parametrize(
    "chain,expected",
    [
        (("scalar-multiply", "invert"), [159, 159, 159]),
        (("invert", "scalar-multiply"), [223, 223, 223]),
    ],
)
def test_transforms_apply_in_chain_order(chain, expected):
    model = _bendable_model()
    by_op = {"scalar-multiply": (2.0,), "invert": (1.0,)}
    transforms = tuple(
        Transform(op, "conv1", by_op[op], _ALL_CHANNELS) for op in chain
    )
    assert _pixel(model.render_frame(render_params(transforms=transforms), 0)) == expected


def test_a_transform_only_touches_the_channels_it_selected():
    model = _bendable_model()
    params = render_params(
        transforms=(Transform("scalar-multiply", "conv1", (3.0,), (0,)),)
    )
    assert _pixel(model.render_frame(params, 0)) == [175, 143, 143]


def test_a_transform_on_the_output_layer_edits_the_final_image():
    model = _bendable_model()
    params = render_params(
        transforms=(Transform("ablate", "output", (1.0,), _ALL_CHANNELS),)
    )
    assert _pixel(model.render_frame(params, 0)) == [128, 128, 128]


def test_a_transform_survives_a_tuple_synthesis_output():
    model = _bendable_model(tuple_output=True)
    params = render_params(
        transforms=(Transform("invert", "output", (1.0,), _ALL_CHANNELS),)
    )
    assert _pixel(model.render_frame(params, 0)) == [239, 239, 239]


def test_an_erode_kernel_size_reaches_the_operator_as_an_int(caplog):
    # torch.ones((1.0, 1.0)) raises, so a float kernel would be logged and
    # skipped instead of applied. Kernel 1 erosion is the identity, which
    # makes "nothing was logged" the whole assertion.
    kernel = manipulation_dict(Transform("erode", "conv1", (5.0,), (0,)), (0,))["params"][0]
    assert kernel == 5 and isinstance(kernel, int)
    model = _bendable_model()
    params = render_params(
        transforms=(Transform("erode", "conv1", (1.0,), _ALL_CHANNELS),)
    )
    with caplog.at_level(logging.WARNING):
        assert _pixel(model.render_frame(params, 0)) == _UNBENT
    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


def test_a_failed_operator_import_is_attempted_only_once(monkeypatch, caplog):
    # A broken bending install (a kornia or torchvision import failure) must
    # not retry the full import machinery every frame: that is exactly the
    # frame rate cost this task exists to make cheap. Patching `__import__`
    # itself, rather than pre-seeding sys.modules, means the count is real
    # regardless of whatever caching Python's own import system would do.
    import builtins

    real_import = builtins.__import__
    attempts = []

    def fake_import(name, *args, **kwargs):
        if name == "autolume.bending.transform_layers":
            attempts.append(name)
            raise ImportError("boom")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    model = _bendable_model()
    params = render_params(
        transforms=(Transform("ablate", "conv1", (1.0,), _ALL_CHANNELS),)
    )
    with caplog.at_level(logging.WARNING):
        for index in range(5):
            assert _pixel(model.render_frame(params, index)) == _UNBENT
    assert len(attempts) == 1
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1


def test_usable_indices_drops_the_channels_an_activation_does_not_have():
    transform = Transform("ablate", "conv1", (1.0,), (0, 2, 5, 99))
    assert usable_indices(transform, 3) == (0, 2)
    assert usable_indices(transform, 100) == (0, 2, 5, 99)
    assert usable_indices(transform, 0) == ()


def test_usable_indices_returns_the_same_tuple_when_nothing_is_out_of_range():
    # The fully in range case is the common one, so it must not allocate an
    # equal copy every frame.
    transform = Transform("ablate", "conv1", (1.0,), (0, 2, 5))
    assert usable_indices(transform, 100) is transform.indices


def test_channels_this_layer_does_not_have_are_dropped_and_logged_once(caplog):
    # Out of range advanced indexing is a device side assert on CUDA, and a
    # poisoned context fails every later frame too, so nothing out of range
    # may reach the operator in the first place. On CPU the caught IndexError
    # would look identical from the outside, so the message is what says the
    # operator was never called rather than called and forgiven.
    model = _bendable_model()
    params = render_params(
        transforms=(Transform("ablate", "conv1", (1.0,), (99,)),)
    )
    with caplog.at_level(logging.WARNING):
        for index in range(3):
            assert _pixel(model.render_frame(params, index)) == _UNBENT
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "does not have" in warnings[0].getMessage()


def test_the_channels_that_do_exist_still_get_bent(caplog):
    model = _bendable_model()
    params = render_params(
        transforms=(Transform("scalar-multiply", "conv1", (3.0,), (0, 99)),)
    )
    with caplog.at_level(logging.WARNING):
        assert _pixel(model.render_frame(params, 0)) == [175, 143, 143]
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "does not have" in warnings[0].getMessage()


def _group_bendable_synthesis():
    """A synthesis whose bendable layer carries a 5D G-CNN activation.

    kornia's geometric operators reject a 5D input outright, which makes this
    the real, device independent way an operator fails at render time. The
    group mean of 0.25 quantizes to 159.
    """
    import torch
    import torch.nn as nn

    class _Group(nn.Module):
        def forward(self, ws):
            return torch.full([1, 3, 2, 1, 2], 0.25)

    class _Synthesis(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = _Group()

        def forward(self, ws, noise_mode="const"):
            return self.conv1(ws).mean(2)

    return _Synthesis()


def test_a_failing_operator_is_skipped_and_logged_once(caplog):
    model = _fake_model(_group_bendable_synthesis())
    params = render_params(
        transforms=(Transform("rotate", "conv1", (10.0,), _ALL_CHANNELS),)
    )
    with caplog.at_level(logging.WARNING):
        for index in range(3):
            assert _pixel(model.render_frame(params, index)) == [159, 159, 159]
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


def test_a_failing_operator_does_not_stop_the_rest_of_the_chain(caplog):
    model = _fake_model(_group_bendable_synthesis())
    params = render_params(
        transforms=(
            Transform("rotate", "conv1", (10.0,), _ALL_CHANNELS),
            Transform("ablate", "conv1", (1.0,), _ALL_CHANNELS),
        )
    )
    with caplog.at_level(logging.WARNING):
        assert _pixel(model.render_frame(params, 0)) == [128, 128, 128]
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


def test_an_operator_failure_with_varying_numbers_is_one_cause(caplog):
    """I1: the dedup key must survive a message that never repeats verbatim.

    A CUDA OOM embeds the live byte count it tried to allocate, so keyed on
    the raw text it filled all 64 `_logged_once` slots in seconds, after
    which every other diagnostic for this model was permanently silent. The
    existing kornia-rejection test cannot see this: its message is constant.
    """
    model = _bendable_model()
    calls = {"n": 0}

    def exploding(tensor, params):
        calls["n"] += 1
        raise RuntimeError(
            f"CUDA out of memory. Tried to allocate {384 + calls['n']} MiB"
        )

    model._manipulation = exploding
    params = render_params(
        transforms=(Transform("ablate", "conv1", (1.0,), _ALL_CHANNELS),)
    )
    with caplog.at_level(logging.WARNING):
        for index in range(3):
            assert _pixel(model.render_frame(params, index)) == _UNBENT
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert len(model._logged_once) == 1


def test_an_operator_failure_with_a_broken_str_does_not_escape_the_hook(caplog):
    """The docstring contract on `_apply_transforms`: it cannot raise.

    The old key stringified the exception unguarded, so a broken `__str__`
    raised inside the forward hook and took the frame down with it.
    """

    class Unprintable(RuntimeError):
        def __str__(self):
            raise ValueError("str() itself blew up")

    def exploding(tensor, params):
        raise Unprintable()

    model = _bendable_model()
    model._manipulation = exploding
    params = render_params(
        transforms=(Transform("ablate", "conv1", (1.0,), _ALL_CHANNELS),)
    )
    with caplog.at_level(logging.WARNING):
        assert _pixel(model.render_frame(params, 0)) == _UNBENT
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "Unprintable" in warnings[0].getMessage()


def test_two_different_failures_are_logged_separately(caplog):
    model = _fake_model(_group_bendable_synthesis())
    params = render_params(
        transforms=(
            Transform("rotate", "conv1", (10.0,), _ALL_CHANNELS),
            Transform("translate", "conv1", (1.0, 1.0), _ALL_CHANNELS),
        )
    )
    with caplog.at_level(logging.WARNING):
        model.render_frame(params, 0)
        model.render_frame(params, 1)
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 2


def test_the_log_once_set_cannot_grow_without_bound(caplog):
    # capture_layer is a free-form string an OSC sender can vary every frame,
    # and it lands in a log key.
    model = _bendable_model()
    with caplog.at_level(logging.WARNING):
        for index in range(_LOG_ONCE_CAP + 5):
            model.render_frame(render_params(capture_layer=f"absent{index}"), index)
    assert len(model._logged_once) == _LOG_ONCE_CAP
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    # One warning per distinct cause, plus exactly one more saying the cap
    # was reached and further distinct causes will not be logged.
    assert len(warnings) == _LOG_ONCE_CAP + 1
    assert "further" in warnings[-1].getMessage()
    assert "not be logged" in warnings[-1].getMessage()


def test_the_log_once_cap_warning_is_itself_logged_only_once(caplog):
    model = _bendable_model()
    with caplog.at_level(logging.WARNING):
        for index in range(_LOG_ONCE_CAP + 20):
            model.render_frame(render_params(capture_layer=f"absent{index}"), index)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    cap_warnings = [r for r in warnings if "further" in r.getMessage()]
    assert len(cap_warnings) == 1


def test_a_layer_name_this_model_does_not_have_is_logged_once_and_skipped(caplog):
    model = _bendable_model()
    params = render_params(
        transforms=(Transform("ablate", "b8.conv0", (1.0,), _ALL_CHANNELS),)
    )
    with caplog.at_level(logging.WARNING):
        for index in range(3):
            assert _pixel(model.render_frame(params, index)) == _UNBENT
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1
    for module in model.G.synthesis.modules():
        assert len(module._forward_hooks) == 0


def test_a_capture_layer_derives_the_image_from_that_layer():
    model = _bendable_model()
    assert _pixel(model.render_frame(render_params(capture_layer="conv1"), 0)) == [
        159,
        159,
        159,
    ]


def test_a_transform_added_to_the_already_captured_layer_still_applies():
    """The capture layer and a transform on it fold into one hook key.

    So adding the session's first transform to the layer being captured does
    not move the key, and anything the hook set up lazily on a key change
    would never happen. This is the ordinary workflow: pick a layer to look
    at, then start bending it.
    """
    model = _bendable_model()
    assert _pixel(model.render_frame(render_params(capture_layer="conv1"), 0)) == [
        159,
        159,
        159,
    ]
    params = render_params(
        capture_layer="conv1",
        transforms=(Transform("invert", "conv1", (1.0,), _ALL_CHANNELS),),
    )
    assert _pixel(model.render_frame(params, 1)) == [223, 223, 223]


def test_a_captured_layer_carries_its_own_transform():
    model = _bendable_model()
    params = render_params(
        capture_layer="conv1",
        transforms=(Transform("invert", "conv1", (1.0,), _ALL_CHANNELS),),
    )
    assert _pixel(model.render_frame(params, 0)) == [223, 223, 223]


def test_a_capture_layer_this_model_does_not_have_falls_back_to_the_final_image(caplog):
    model = _bendable_model()
    params = render_params(capture_layer="b8.torgb")
    with caplog.at_level(logging.WARNING):
        for index in range(3):
            assert _pixel(model.render_frame(params, index)) == _UNBENT
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


def test_capturing_a_group_layer_averages_the_group_dimension():
    import torch
    import torch.nn as nn

    class _GroupBlock(nn.Module):
        def forward(self, ws):
            # (N, C, G, H, W) with two groups, 0.0 and 0.5, so the group mean
            # is 0.25 and quantizes to 159.
            groups = torch.tensor([0.0, 0.5]).reshape(1, 1, 2, 1, 1)
            return groups.repeat(1, 3, 1, 1, 2)

    class _Synthesis(nn.Module):
        def __init__(self):
            super().__init__()
            self.group_conv = _GroupBlock()

        def forward(self, ws, noise_mode="const"):
            self.group_conv(ws)
            return torch.zeros([1, 3, 1, 2])

    model = _fake_model(_Synthesis())
    frame = model.render_frame(render_params(capture_layer="group_conv"), 0)
    assert frame.shape == (1, 2, 3)
    assert _pixel(frame) == [159, 159, 159]


def test_the_frame_slot_does_not_outlive_the_frame():
    model = _bendable_model()
    params = render_params(
        transforms=(Transform("ablate", "conv1", (1.0,), _ALL_CHANNELS),)
    )
    assert _pixel(model.render_frame(params, 0)) == [128, 128, 128]
    # The hooks stay registered, so a synthesis call from anywhere else would
    # be bent too if the per frame slot were left behind.
    loose = model.G.synthesis(None)
    assert float(loose.reshape(-1)[0]) == 0.125


def test_the_frame_slot_is_cleared_even_when_synthesis_raises():
    import torch
    import torch.nn as nn

    class _Flaky(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Identity()
            self.calls = 0

        def forward(self, ws, noise_mode="const"):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("synthesis exploded")
            return self.conv1(torch.full([1, 3, 1, 2], 0.25))

    model = _fake_model(_Flaky())
    params = render_params(
        transforms=(Transform("ablate", "conv1", (1.0,), _ALL_CHANNELS),)
    )
    with pytest.raises(RuntimeError):
        model.render_frame(params, 0)
    loose = model.G.synthesis(None)
    assert float(loose.reshape(-1)[0]) == 0.25


def test_a_layer_with_nothing_to_bend_is_left_alone():
    import torch
    import torch.nn as nn

    class _Empty(nn.Module):
        def forward(self, x):
            return ()

    class _Synthesis(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = _Empty()

        def forward(self, ws, noise_mode="const"):
            self.conv1(ws)
            return torch.full([1, 3, 1, 2], 0.25)

    model = _fake_model(_Synthesis())
    params = render_params(
        capture_layer="conv1",
        transforms=(Transform("ablate", "conv1", (1.0,), _ALL_CHANNELS),),
    )
    assert _pixel(model.render_frame(params, 0)) == [159, 159, 159]


def test_release_removes_hook_handles_and_leaves_the_module_clean():
    """The reference cycle this task's brief flags: `G ->
    module._forward_hooks -> closure -> LoadedModel -> G`. Removing the
    handles is what breaks it, so this checks the module's own hook table,
    not just `LoadedModel`'s bookkeeping list. Single threaded: it verifies
    the cycle is broken structurally, nothing about the render/loader
    thread interleaving around a real retirement, which is a separate
    question this test does not answer."""
    model = _bendable_model()
    params = render_params(
        transforms=(Transform("ablate", "conv1", (1.0,), _ALL_CHANNELS),)
    )
    model.render_frame(params, 0)
    assert len(model._hook_handles) == 1
    assert model.G.synthesis.conv1.hook_registrations == 1

    model.release()

    assert model._hook_handles == []
    for module in model.G.synthesis.modules():
        assert len(module._forward_hooks) == 0

    # Idempotent: a second release on an already-released model is safe.
    model.release()
    assert model._hook_handles == []


# --- network mixing (slot B, the mixed network, saving) -------------------


def wait_for(predicate, timeout=5.0):
    """Poll `predicate` until it holds, then return whether it did."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def tiny_generator(seed=0, img_resolution=16, channel_max=8):
    """A real custom stylegan2 generator, small enough to build in tests.

    Every parameter and stateful buffer is perturbed so no tensor is bit
    identical to another model's. A freshly built generator zero-initialises
    over half its tensors (every bias, every affine bias, every noise
    strength, `w_avg`), and a pair of them would let a merge taking the
    wrong source pass most of its assertions. See
    `tests/live/test_mixing.py::randomize` for why this perturbs rather than
    redraws, and why `resample_filter` is left alone.

    `synthesis_kwargs` is passed explicitly because `Generator.__init__`
    declares it as a mutable default and updates it in place.
    """
    import torch

    from architectures import custom_stylegan2

    torch.manual_seed(seed)
    model = custom_stylegan2.Generator(
        z_dim=8,
        c_dim=0,
        w_dim=8,
        img_channels=3,
        img_resolution=img_resolution,
        synthesis_kwargs={"channel_base": 64, "channel_max": channel_max},
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.1 + seed * 0.02)
        for name, buffer in model.named_buffers():
            if not name.endswith("resample_filter"):
                buffer.add_(torch.randn_like(buffer) * 0.1 + seed * 0.02)
    return model


def generator_loader(by_path):
    """A loader handing back prebuilt generators wrapped in `LoadedModel`."""
    import torch

    def loader(path, device=None):
        return LoadedModel(path, by_path[path], device or torch.device("cpu"))

    return loader


def mixing_host(a, b, entries=None):
    """A started host with `a` in slot A, `b` in slot B and mixing on.

    Returns the host once both slots are filled; the caller decides what
    selection to send.
    """
    host = ModelHost(
        loader=generator_loader({"/tmp/a.pkl": a, "/tmp/b.pkl": b})
    )
    host.request_load("/tmp/a.pkl")
    host.request_load_b("/tmp/b.pkl")
    assert wait_for(lambda: host.current() is not None and host.current_b() is not None)
    if entries is not None:
        host.set_mixing_enabled(True)
        host.request_mix(entries)
    return host


def split_at_resolution(a, boundary):
    """A selection taking every layer up to `boundary` from A, the rest from B."""
    return [
        "A" if layer_resolution(name) <= boundary else "B"
        for name in conv_names(a)
    ]


def test_model_host_loads_slot_b_without_disturbing_slot_a():
    a, b = tiny_generator(seed=1), tiny_generator(seed=2)
    host = ModelHost(loader=generator_loader({"/tmp/a.pkl": a, "/tmp/b.pkl": b}))
    host.request_load("/tmp/a.pkl")
    assert wait_for(lambda: host.current() is not None)
    rendering = host.current()

    host.request_load_b("/tmp/b.pkl")
    assert wait_for(lambda: host.current_b() is not None)
    assert host.current_b().G is b
    assert host.current() is rendering
    assert host.error() is None
    host.stop()


def test_model_host_keeps_slot_b_on_the_cpu():
    """Slot B is a weight source, never a rendered model, so it never takes
    room on the render device and never needs re-homing on a device switch."""
    import torch

    seen = []

    def loader(path, device=None):
        seen.append((path, device))
        return LoadedModel(path, tiny_generator(), device or torch.device("cpu"))

    host = ModelHost(loader=loader)
    host.request_load_b("/tmp/b.pkl")
    assert wait_for(lambda: host.current_b() is not None)
    assert seen == [("/tmp/b.pkl", torch.device("cpu"))]
    host.stop()


def test_model_host_reports_a_slot_b_load_failure():
    def loader(path, device=None):
        raise RuntimeError("bad second pkl")

    host = ModelHost(loader=loader)
    host.request_load_b("/tmp/b.pkl")
    assert wait_for(lambda: host.error() is not None)
    assert "bad second pkl" in host.error()
    assert host.current_b() is None
    assert host.pending_b() is None
    host.stop()


def test_model_host_a_slot_b_load_does_not_clear_model_as_error():
    """Slot B succeeding says nothing about a slot A that is still not
    loaded, and the preview overlay must keep saying so."""
    def loader(path, device=None):
        if path == "/tmp/a.pkl":
            raise RuntimeError("bad first pkl")
        return LoadedModel(path, tiny_generator(), device)

    host = ModelHost(loader=loader)
    host.request_load("/tmp/a.pkl")
    assert wait_for(lambda: host.error() is not None)

    host.request_load_b("/tmp/b.pkl")
    assert wait_for(lambda: host.current_b() is not None)
    assert "bad first pkl" in host.error()
    assert host.current() is None
    host.stop()


def test_model_host_renders_the_mix_once_it_is_built():
    a, b = tiny_generator(seed=1), tiny_generator(seed=2)
    host = mixing_host(a, b, split_at_resolution(a, 8))
    assert wait_for(lambda: host.current().G is not a)

    mixed = host.current()
    assert mixed.G is not a and mixed.G is not b
    # Both sources stay loaded and untouched.
    assert host.current_b().G is b
    assert host.error() is None
    state_a, state_b, state_mixed = a.state_dict(), b.state_dict(), mixed.G.state_dict()
    for name in conv_names(a):
        source = state_a if layer_resolution(name) <= 8 else state_b
        assert mixed.G.state_dict()[name].shape == source[name].shape
        assert (state_mixed[name].cpu() == source[name].cpu()).all(), name
    host.stop()


def test_model_host_renders_model_a_while_mixing_is_off():
    a, b = tiny_generator(seed=1), tiny_generator(seed=2)
    host = mixing_host(a, b, split_at_resolution(a, 8))
    assert wait_for(lambda: host.current().G is not a)

    host.set_mixing_enabled(False)
    assert host.current().G is a
    host.stop()


def test_model_host_toggling_mixing_back_on_reuses_the_built_mix():
    a, b = tiny_generator(seed=1), tiny_generator(seed=2)
    host = mixing_host(a, b, split_at_resolution(a, 8))
    assert wait_for(lambda: host.current().G is not a)
    mixed = host.current()

    host.set_mixing_enabled(False)
    host.set_mixing_enabled(True)
    assert host.current() is mixed
    host.stop()


def test_current_a_is_slot_a_even_while_the_mix_is_what_renders():
    """`current()` answers what is on screen; `current_a()` answers which model
    the mixing selection applies to. While a mix renders they are different
    networks, and a caller that wants model A's layer names has to have a way to
    say so.
    """
    a, b = tiny_generator(seed=1), tiny_generator(seed=2)
    host = mixing_host(a, b, split_at_resolution(a, 8))
    assert wait_for(lambda: host.current().G is not a)

    assert host.current().G is not a
    assert host.current_a().G is a
    host.stop()


def test_retiring_a_mix_leaves_mixing_enabled_set():
    """Pins the fact that made a `mixing_enabled()` gate the wrong test.

    A model A swap retires the mix, so `current()` goes back to model A, but
    `_mixing_enabled` is deliberately left on: the flag is the performer's
    intent, not a statement about what is built. Anything gating a slot A read
    on it therefore stays shut for good after the first mix, which is exactly
    what `current_a()` exists to avoid.
    """
    a, b = tiny_generator(seed=1), tiny_generator(seed=2)
    host = mixing_host(a, b, split_at_resolution(a, 8))
    assert wait_for(lambda: host.current().G is not a)

    replacement = tiny_generator(seed=3)
    host._loader = generator_loader({"/tmp/c.pkl": replacement})
    host.request_load("/tmp/c.pkl")
    assert wait_for(lambda: host.current_a().G is replacement)

    assert host.mixing_enabled() is True
    assert host.current_a().G is replacement
    host.stop()


def test_model_host_a_failed_mix_keeps_rendering_a_and_reports_it():
    # Different block widths split at a boundary: the pair cannot assemble.
    a = tiny_generator(seed=1, channel_max=8)
    b = tiny_generator(seed=2, channel_max=16)
    host = mixing_host(a, b, split_at_resolution(a, 4))
    assert wait_for(lambda: host.error() is not None)

    assert host.error() == INCOMPATIBLE_MODELS
    assert host.current().G is a
    assert host.mixing_enabled() is True
    host.stop()


def test_model_host_a_leading_x_is_refused_and_reported():
    """The refusal has to reach the performer, not just the log.

    A selection whose first entry is neither model used to assemble a
    generator with a freshly random mapping network and render it as if it
    were a mix. The check lives in `combine` rather than in the watcher
    precisely so it comes out here, through the same `_drop_mix` path every
    other mixing failure uses, with model A still on screen.
    """
    a, b = tiny_generator(seed=1), tiny_generator(seed=2)
    entries = ["X"] + ["A"] * (selection_length(a, b) - 1)
    host = mixing_host(a, b, entries)
    assert wait_for(lambda: host.error() is not None)

    assert "first layer" in host.error()
    assert host.current().G is a
    host.stop()


def test_model_host_a_selection_of_the_wrong_length_keeps_rendering_a():
    a, b = tiny_generator(seed=1), tiny_generator(seed=2)
    host = mixing_host(a, b, ["A"] * (selection_length(a, b) - 1))
    assert wait_for(lambda: host.error() is not None)

    assert "entries" in host.error()
    assert host.current().G is a
    host.stop()


def test_model_host_a_mix_without_a_second_model_keeps_rendering_a():
    a = tiny_generator(seed=1)
    host = ModelHost(loader=generator_loader({"/tmp/a.pkl": a}))
    host.request_load("/tmp/a.pkl")
    assert wait_for(lambda: host.current() is not None)

    host.set_mixing_enabled(True)
    host.request_mix(["A"] * len(conv_names(a)))
    assert wait_for(lambda: host.error() is not None)
    assert "both slots" in host.error()
    assert host.current().G is a
    host.stop()


def test_model_host_an_empty_selection_drops_the_mix():
    a, b = tiny_generator(seed=1), tiny_generator(seed=2)
    host = mixing_host(a, b, split_at_resolution(a, 8))
    assert wait_for(lambda: host.current().G is not a)

    host.request_mix(())
    assert wait_for(lambda: host.current().G is a)
    assert host.error() is None
    host.stop()


def test_model_host_publishes_the_mixed_models_layer_catalog():
    a, b = tiny_generator(seed=1), tiny_generator(seed=2)
    host = mixing_host(a, b)
    plain = host.info_store.snapshot()
    assert plain.layers and plain.layers[-1].width == 16

    truncating = ["X" if layer_resolution(n) == 16 else "A" for n in conv_names(a)]
    host.set_mixing_enabled(True)
    host.request_mix(truncating)
    assert wait_for(lambda: host.info_store.snapshot() is not plain)

    mixed = host.info_store.snapshot()
    assert mixed.layers and mixed.layers[-1].width == 8
    host.stop()


def test_model_host_republishes_model_a_catalog_when_mixing_turns_off():
    a, b = tiny_generator(seed=1), tiny_generator(seed=2)
    host = mixing_host(a, b)
    plain = host.info_store.snapshot()

    truncating = ["X" if layer_resolution(n) == 16 else "A" for n in conv_names(a)]
    host.set_mixing_enabled(True)
    host.request_mix(truncating)
    assert wait_for(lambda: host.info_store.snapshot() is not plain)

    host.set_mixing_enabled(False)
    assert host.info_store.snapshot() == plain
    host.stop()


def test_model_host_a_mix_that_cannot_be_enumerated_still_renders(monkeypatch):
    """A mixed generator whose catalog cannot be built is still a perfectly
    good network to render. `_model_info`'s guard has to hold for it, or a
    failed enumeration would take the whole mix down with it."""
    a, b = tiny_generator(seed=1), tiny_generator(seed=2)
    host = mixing_host(a, b)

    def boom(self):
        raise RuntimeError("no catalog here")

    monkeypatch.setattr(LoadedModel, "enumerate_layers", boom)
    host.set_mixing_enabled(True)
    host.request_mix(split_at_resolution(a, 8))
    assert wait_for(lambda: host.current().G is not a)

    assert host.info_store.snapshot().layers == ()
    assert host.error() is None
    host.stop()


def test_model_host_a_new_model_a_retires_the_mix_and_rebuilds_it():
    a, b = tiny_generator(seed=1), tiny_generator(seed=2)
    other = tiny_generator(seed=3)
    host = ModelHost(
        loader=generator_loader(
            {"/tmp/a.pkl": a, "/tmp/b.pkl": b, "/tmp/other.pkl": other}
        )
    )
    host.request_load("/tmp/a.pkl")
    host.request_load_b("/tmp/b.pkl")
    assert wait_for(lambda: host.current() is not None and host.current_b() is not None)
    host.set_mixing_enabled(True)
    host.request_mix(["A"] * selection_length(a, b))
    assert wait_for(lambda: host.current().G is not a)
    first_mix = host.current()

    host.request_load("/tmp/other.pkl")
    # `is not other` is what tells a rebuilt mix apart from the mix simply
    # being dropped and the new slot A rendering bare.
    assert wait_for(
        lambda: host.current() is not first_mix and host.current().G is not other
    )
    rebuilt = host.current()
    assert rebuilt.G is not a
    name = conv_names(other)[1]
    assert (
        rebuilt.G.state_dict()[name] == other.state_dict()[name]
    ).all()
    host.stop()


def test_model_host_a_device_switch_retires_the_mix_and_rebuilds_it():
    """A mix is pinned to the device slot A had when it was built, so a
    device switch has to throw it away rather than keep rendering a network
    sitting on the device the runtime just left."""
    a, b = tiny_generator(seed=1), tiny_generator(seed=2)
    host = mixing_host(a, b, ["A"] * selection_length(a, b))
    assert wait_for(lambda: host.current().G is not a)
    first_mix = host.current()

    host.request_device("cpu")
    assert wait_for(
        lambda: host.current() is not first_mix and host.current().G is not a
    )
    rebuilt = host.current()
    assert next(rebuilt.G.parameters()).device == rebuilt.device
    host.stop()


def test_model_host_a_new_model_b_retires_the_mix_and_rebuilds_it():
    a, b = tiny_generator(seed=1), tiny_generator(seed=2)
    other = tiny_generator(seed=3)
    host = ModelHost(
        loader=generator_loader(
            {"/tmp/a.pkl": a, "/tmp/b.pkl": b, "/tmp/other.pkl": other}
        )
    )
    host.request_load("/tmp/a.pkl")
    host.request_load_b("/tmp/b.pkl")
    assert wait_for(lambda: host.current() is not None and host.current_b() is not None)
    host.set_mixing_enabled(True)
    host.request_mix(["B"] * selection_length(a, b))
    assert wait_for(lambda: host.current().G is not a)
    first_mix = host.current()

    host.request_load_b("/tmp/other.pkl")
    # `is not a` as well as `is not first_mix`: retiring the mix makes
    # `current()` fall back to bare model A for the moment before the
    # rebuild lands, and waiting only on `is not first_mix` would accept
    # that intermediate state and then compare A against `other`.
    assert wait_for(
        lambda: host.current() is not first_mix and host.current().G is not a
    )
    name = conv_names(other)[1]
    assert (
        host.current().G.state_dict()[name] == other.state_dict()[name]
    ).all()
    host.stop()


def test_model_host_releasing_the_mix_leaves_both_sources_usable():
    """A mix holds copies, never the sources' own tensors, so retiring it
    can never reach back into either model it was built from."""
    a, b = tiny_generator(seed=1), tiny_generator(seed=2)
    host = mixing_host(a, b, ["A"] * selection_length(a, b))
    assert wait_for(lambda: host.current().G is not a)
    name = conv_names(a)[1]
    before = a.state_dict()[name].clone()

    host.request_mix(())
    assert wait_for(lambda: host.current().G is a)
    assert (a.state_dict()[name] == before).all()
    assert host.current_b().G is b
    host.stop()


def test_request_mix_ignores_a_value_that_is_not_a_sequence(caplog):
    host = ModelHost(loader=FakeModel)
    with caplog.at_level(logging.WARNING):
        host.request_mix(7)
    assert any("not a sequence" in r.getMessage() for r in caplog.records)
    host.stop()


def use_data_root(monkeypatch, root):
    from utils import user_data

    monkeypatch.setattr(user_data, "_prefs", {"version": 1, "data_root": str(root)})
    monkeypatch.setattr(user_data, "_data_root", str(root))


def stub_discriminator(monkeypatch):
    import torch

    import autolume.live.core.generator as generator_module

    discriminator = torch.nn.Linear(2, 2)
    monkeypatch.setattr(
        generator_module, "load_discriminator", lambda path: discriminator
    )
    return discriminator


def test_model_host_saves_the_merged_model_next_to_the_users_models(
    monkeypatch, tmp_path
):
    import pickle

    use_data_root(monkeypatch, tmp_path)
    stub_discriminator(monkeypatch)
    a, b = tiny_generator(seed=1), tiny_generator(seed=2)
    host = mixing_host(a, b, split_at_resolution(a, 8))
    assert wait_for(lambda: host.current().G is not a)

    host.request_save_mix("merged")
    assert wait_for(lambda: host.mix_save_store.snapshot().path is not None)

    status = host.mix_save_store.snapshot()
    assert status.error is None
    assert status.path == str(tmp_path / "models" / "merged.pkl")
    with open(status.path, "rb") as handle:
        data = pickle.load(handle)
    assert set(data) == {"G", "G_ema", "D"}
    assert data["G"] is data["G_ema"]

    # The file has to be the mix that was on screen. `_save_mix` assembles a
    # second time, so anything the merge draws fresh instead of copying
    # would make the saved model quietly different from the preview.
    import torch

    rendering = host.current().G.state_dict()
    saved = data["G_ema"].state_dict()
    assert set(saved) == set(rendering)
    for name in rendering:
        assert torch.equal(saved[name], rendering[name].cpu()), name
    host.stop()


def test_a_second_save_requested_mid_build_still_writes_its_file(
    monkeypatch, tmp_path
):
    """I4: `_run` snapshots its work and then spends seconds on loads and
    builds before `_save_mix` runs, so a Save requested in that window used
    to be cleared unconditionally: no file, no error, and a green line
    naming the first file. The performer who retypes a wrong name and
    clicks Save again mid merge must get their second file.

    The choreography holds the loader inside slot-B loads so the second
    Save deterministically lands after the cycle snapshotted the first."""
    use_data_root(monkeypatch, tmp_path)
    stub_discriminator(monkeypatch)
    a, b = tiny_generator(seed=1), tiny_generator(seed=2)
    inner = generator_loader(
        {"/tmp/a.pkl": a, "/tmp/b.pkl": b, "/tmp/b2.pkl": b, "/tmp/b3.pkl": b}
    )
    gates = {"/tmp/b2.pkl": threading.Event(), "/tmp/b3.pkl": threading.Event()}
    entered = {"/tmp/b2.pkl": threading.Event(), "/tmp/b3.pkl": threading.Event()}

    def loader(path, device=None):
        gate = gates.get(path)
        if gate is not None:
            entered[path].set()
            gate.wait(5.0)
        return inner(path, device)

    host = ModelHost(loader=loader)
    try:
        host.request_load("/tmp/a.pkl")
        host.request_load_b("/tmp/b.pkl")
        assert wait_for(
            lambda: host.current() is not None and host.current_b() is not None
        )
        host.set_mixing_enabled(True)
        host.request_mix(split_at_resolution(a, 8))
        assert wait_for(lambda: host.current().G is not a)

        # A cycle snapshots (load b2), blocks inside the loader; the first
        # Save and a further B load queue up behind it.
        host.request_load_b("/tmp/b2.pkl")
        assert entered["/tmp/b2.pkl"].wait(5.0)
        host.request_save_mix("first")
        host.request_load_b("/tmp/b3.pkl")
        gates["/tmp/b2.pkl"].set()

        # The next cycle snapshots save_name="first" and blocks in the b3
        # load: the second Save lands exactly mid build.
        assert entered["/tmp/b3.pkl"].wait(5.0)
        host.request_save_mix("second")
        gates["/tmp/b3.pkl"].set()

        assert wait_for(lambda: (tmp_path / "models" / "first.pkl").exists())
        assert wait_for(lambda: (tmp_path / "models" / "second.pkl").exists())
        assert wait_for(
            lambda: host.mix_save_store.snapshot().path
            == str(tmp_path / "models" / "second.pkl")
        )
    finally:
        for gate in gates.values():
            gate.set()
        host.stop()


def test_model_host_a_save_name_cannot_escape_the_models_folder(
    monkeypatch, tmp_path
):
    use_data_root(monkeypatch, tmp_path)
    stub_discriminator(monkeypatch)
    a, b = tiny_generator(seed=1), tiny_generator(seed=2)
    host = mixing_host(a, b, split_at_resolution(a, 8))
    assert wait_for(lambda: host.current().G is not a)

    host.request_save_mix("../../escaped.pkl")
    assert wait_for(lambda: host.mix_save_store.snapshot().path is not None)

    assert host.mix_save_store.snapshot().path == str(
        tmp_path / "models" / "escaped.pkl"
    )
    host.stop()


def test_model_host_a_save_without_a_second_model_reports_and_writes_nothing(
    monkeypatch, tmp_path
):
    use_data_root(monkeypatch, tmp_path)
    a = tiny_generator(seed=1)
    host = ModelHost(loader=generator_loader({"/tmp/a.pkl": a}))
    host.request_load("/tmp/a.pkl")
    assert wait_for(lambda: host.current() is not None)

    host.request_save_mix("merged")
    assert wait_for(lambda: host.mix_save_store.snapshot().error is not None)

    status = host.mix_save_store.snapshot()
    assert status.path is None
    assert "both slots" in status.error
    # A save failure is the mixing panel's news, never the preview's.
    assert host.error() is None
    assert not (tmp_path / "models" / "merged.pkl").exists()
    host.stop()


def test_model_host_a_save_with_no_name_reports_rather_than_writing(
    monkeypatch, tmp_path
):
    use_data_root(monkeypatch, tmp_path)
    a, b = tiny_generator(seed=1), tiny_generator(seed=2)
    host = mixing_host(a, b, split_at_resolution(a, 8))
    assert wait_for(lambda: host.current().G is not a)

    host.request_save_mix("   ")
    assert wait_for(lambda: host.mix_save_store.snapshot().error is not None)
    assert "file name" in host.mix_save_store.snapshot().error
    host.stop()


def test_mix_save_status_starts_empty():
    host = ModelHost(loader=FakeModel)
    assert host.mix_save_store.snapshot() == MixSaveStatus()
    host.stop()


def test_request_mix_ignores_a_bare_string(caplog):
    host = ModelHost(loader=FakeModel)
    with caplog.at_level(logging.WARNING):
        host.request_mix("ABX")
    assert any("not a sequence" in r.getMessage() for r in caplog.records)
    host.stop()


def test_model_host_an_all_a_mix_renders_the_same_frame_as_model_a():
    """The plan's acceptance criterion at the far end of the pipeline:
    equal pixels through the render path, not just equal tensors.

    At the registry defaults, so truncation (0.8, which reads `w_avg`) and
    the const noise mode (which reads every `noise_const`) are both live.
    Those are exactly the two buffers a parameters-only merge would leave
    freshly constructed, and both are what this asserts against.
    """
    a, b = tiny_generator(seed=1), tiny_generator(seed=2)
    host = mixing_host(a, b)
    params = render_params()
    plain = host.current().render_frame(params, 0)

    host.set_mixing_enabled(True)
    host.request_mix(["A"] * selection_length(a, b))
    assert wait_for(lambda: host.current().G is not a)
    mixed = host.current().render_frame(params, 0)

    assert mixed.shape == plain.shape
    assert (mixed == plain).all()
    host.stop()


def test_model_host_a_split_mix_renders_a_different_frame_from_model_a():
    """Guards the test above from passing on a mix that is silently A."""
    a, b = tiny_generator(seed=1), tiny_generator(seed=2)
    host = mixing_host(a, b)
    params = render_params()
    plain = host.current().render_frame(params, 0)

    host.set_mixing_enabled(True)
    host.request_mix(split_at_resolution(a, 8))
    assert wait_for(lambda: host.current().G is not a)
    mixed = host.current().render_frame(params, 0)

    assert mixed.shape == plain.shape
    assert not (mixed == plain).all()
    host.stop()
