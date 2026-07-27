"""Model loading, latent navigation, and synthesis.

All torch-and-model specifics live here so the render loop stays
orchestration-only. Latent navigation is the seed-grid bilinear walk:
every integer point of seed space owns a deterministic z draw, and a
continuous position blends the four surrounding seeds in w space.
Ported from balagan (latent_navigator.py).
"""

import logging
import math
import os
import threading
from dataclasses import dataclass
from typing import Callable

import numpy as np

from autolume.live.core.mixing import combine
from autolume.live.core.params import Keyframe, RenderParams, Transform
from autolume.live.core.store import LatestValueStore
from autolume.live.core.superres import SuperRes

logger = logging.getLogger(__name__)

_SEED_MASK = (1 << 32) - 1
_BILINEAR_CORNERS = ((0, 0), (1, 0), (0, 1), (1, 1))
_SLERP_COLINEAR_THRESHOLD = 0.9995
_KEYFRAME_CACHE_SIZE = 4
_FRAME_CHANNELS = 3
# An all zero channel has no peak to divide by, and 0/0 is NaN. Anything with
# a peak at all normalizes normally, floor or no floor.
_NORMALIZE_FLOOR = 1e-8
# Distinct causes worth one warning each. Some log keys carry a layer name
# straight from the params, and `capture_layer` is a free-form string an OSC
# sender can vary every frame, so the set needs a ceiling.
_LOG_ONCE_CAP = 64


def slerp(alpha: float, w0, w1):
    """Spherical interpolation between two W tensors.

    Falls back to lerp when the vectors are close enough to colinear that the
    angle between them is not numerically well defined.
    """
    import torch

    dot = (w0 * w1).sum() / (w0.norm() * w1.norm() + 1e-12)
    dot = dot.clamp(-1.0, 1.0)
    if dot.abs() > _SLERP_COLINEAR_THRESHOLD:
        return w0 + alpha * (w1 - w0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    return (
        torch.sin((1.0 - alpha) * theta) * w0 + torch.sin(alpha * theta) * w1
    ) / sin_theta


def corner_seeds(
    latent_x: float, latent_y: float, step_y: int = 100
) -> list[tuple[int, float]]:
    base_x = math.floor(latent_x)
    base_y = math.floor(latent_y)
    corners: list[tuple[int, float]] = []
    for offset_x, offset_y in _BILINEAR_CORNERS:
        seed_x = base_x + offset_x
        seed_y = base_y + offset_y
        seed = (seed_x + seed_y * step_y) & _SEED_MASK
        weight = (1.0 - abs(latent_x - seed_x)) * (1.0 - abs(latent_y - seed_y))
        if weight > 0.0:
            corners.append((seed, weight))
    return corners


def noise_mode(params: RenderParams) -> str:
    """Map the noise parameters onto the synthesis `noise_mode` argument.

    Seed 0 means the model's own constant noise buffer, so composition stays
    put while any other seed redraws the texture. Animation forces "random"
    because the constant buffer cannot animate.
    """
    if not params.noise_enabled:
        return "none"
    if params.noise_anim or params.noise_seed != 0:
        return "random"
    return "const"


def effective_noise_seed(params: RenderParams, frame_index: int) -> int:
    seed = params.noise_seed
    if params.noise_anim:
        seed += frame_index
    return seed & _SEED_MASK


def hook_layer_names(params: RenderParams) -> tuple[str, ...]:
    """Every layer this frame reads or edits, in first seen order.

    This tuple is the hook registration key: while it holds, the registered
    hooks are reused frame after frame, however much the transforms
    themselves are being turned.
    """
    names: list[str] = []
    for transform in params.transforms:
        if transform.layer not in names:
            names.append(transform.layer)
    if params.capture_layer and params.capture_layer not in names:
        names.append(params.capture_layer)
    return tuple(names)


def usable_indices(transform: Transform, channels: int) -> tuple[int, ...]:
    """The transform's selected channels that this activation actually has.

    Selections are authored against one model and then persist through
    presets and model swaps, so a chain built on a 512 channel layer can
    arrive at a 128 channel one. Every operator reaches its tensor through
    `x[:, indices]`, and out of range advanced indexing is a device side
    assert on CUDA: catchable, but it poisons the context, so every frame
    after it fails too. Filtering here is the only place that knows the
    tensor, and it is the difference between one skipped transform and a
    dead render thread.

    The common case is every index already in range, so the fully-in-range
    path returns `transform.indices` itself rather than building an equal
    copy every frame.
    """
    indices = transform.indices
    if all(0 <= index < channels for index in indices):
        return indices
    return tuple(index for index in indices if 0 <= index < channels)


def manipulation_dict(transform: Transform, indices) -> dict:
    """A `Transform` in the shape `ManipulationLayer.forward` reads.

    `indices` is passed separately because the caller has already bounded it
    against the activation. `erode` and `dilate` size a `torch.ones` kernel
    with their parameter, so the stored float has to become an int on the way
    in or the kernel cannot be built at all.
    """
    params = [float(value) for value in transform.params]
    if transform.op in ("erode", "dilate") and params:
        params[0] = int(params[0])
    return {
        "transformID": transform.op,
        "params": params,
        "indices": list(indices),
    }


def adjust_weights(params: RenderParams) -> tuple[float, ...]:
    """The eight adjuster weights in slot order."""
    return (
        params.adjust_w1,
        params.adjust_w2,
        params.adjust_w3,
        params.adjust_w4,
        params.adjust_w5,
        params.adjust_w6,
        params.adjust_w7,
        params.adjust_w8,
    )


def direction_delta(
    params: RenderParams, width: int
) -> tuple[np.ndarray | None, tuple[int, ...]]:
    """The adjuster's weighted direction sum, and the slots that could not join.

    None means there is nothing to add to W: every weight is zero, no
    direction is loaded, or the only weighted directions are the wrong
    width for this model. A direction of the wrong width is reported back
    for the caller to log rather than padded or truncated into the sum.
    Zero-weighted slots are never checked, so a stale direction from
    another model stays silent until somebody actually asks for it.
    """
    delta = None
    mismatched: list[int] = []
    for index, (weight, direction) in enumerate(
        zip(adjust_weights(params), params.directions)
    ):
        if weight == 0.0:
            continue
        if len(direction) != width:
            mismatched.append(index)
            continue
        term = np.array(direction, dtype=np.float32) * np.float32(weight)
        delta = term if delta is None else delta + term
    return delta, tuple(mismatched)


def channel_window(activation, base_channel: int, grayscale: bool):
    """The three channel window a `[C, H, W]` activation is shown through.

    Grayscale reads one channel and replicates it; colour reads three
    consecutive channels, falling back to one when the activation is
    narrower than three. `base_channel` is clamped so the window always
    lands inside the activation, whatever model the number was chosen on.
    """
    channels = int(activation.shape[0])
    count = 1 if grayscale else _FRAME_CHANNELS
    if count > channels:
        count = 1
    base = max(0, min(int(base_channel), channels - count))
    window = activation[base : base + count]
    if window.shape[0] == 1:
        window = window.repeat(_FRAME_CHANNELS, 1, 1)
    return window


def derive_float_image(activation, params: RenderParams):
    """The float image a frame is derived from, before uint8 conversion.

    Window, then normalize, then scale, which is the order the old app
    used. Normalization is scale invariant, so scaling first would swallow
    the decibel gain.
    """
    import torch

    image = channel_window(
        activation, params.base_channel, params.grayscale
    ).to(torch.float32)
    if params.img_normalize:
        peak = image.abs().amax(dim=(1, 2), keepdim=True).clamp(min=_NORMALIZE_FLOOR)
        image = image / peak
    if params.img_scale_db:
        image = image * (10.0 ** (params.img_scale_db / 20.0))
    return image


def to_uint8_frame(activation) -> np.ndarray:
    """A `[3, H, W]` float image to the contiguous HWC uint8 frame."""
    import torch

    frame = (activation * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return frame.permute(1, 2, 0).contiguous().cpu().numpy()


def pick_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DeviceUnavailable(Exception):
    """A requested device name this machine cannot actually provide.

    Raised by `resolve_device`, on the loader thread, for `cuda`/`mps` when
    that backend is not available, or for a name outside the registry's four
    valid values. Kept distinct from a plain load failure so the caller can
    tell "this pkl is broken" from "this device does not exist here" without
    parsing an error string.
    """


def resolve_device(name: str):
    """A `device` registry value (`auto`/`cuda`/`mps`/`cpu`) to a real device.

    `auto` always resolves, through `pick_device`, the same as an initial
    model load with no device requested. Every other name is validated
    against what this machine actually has, so a switch to an unavailable
    device fails here rather than partway through `.to(device)`.
    """
    import torch

    if name == "auto":
        return pick_device()
    if name == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise DeviceUnavailable("CUDA is not available on this machine")
    if name == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise DeviceUnavailable("MPS is not available on this machine")
    if name == "cpu":
        return torch.device("cpu")
    raise DeviceUnavailable(f"Unknown device {name!r}")


@dataclass(frozen=True)
class DeviceStatus:
    """What the render device is doing, published for the performance panel.

    `active` is the device string the currently loaded model actually runs
    on (None before anything has loaded), never what was merely asked for.
    `requested` is the device name the most recent switch attempt was made
    with, so a watcher can tell which attempt a given status answers.
    `error` is set only when that attempt failed, and cleared by the next
    one that succeeds.
    """

    active: str | None = None
    requested: str = "auto"
    error: str | None = None


@dataclass(frozen=True)
class MixSaveStatus:
    """The outcome of the most recent save-merged-model job.

    Kept off the `error()` channel the preview overlay reads: a save that
    failed is news for the mixing panel that asked for it, not a reason to
    tell the performer their model is broken while it renders perfectly.
    """

    path: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class LayerInfo:
    """One synthesis submodule's output shape, as seen by the layer catalog."""

    name: str
    channels: int
    width: int
    height: int


@dataclass(frozen=True)
class ModelInfo:
    """Immutable dimensions of a loaded model, published on the loader thread.

    The control plane needs `z_dim` to materialize latent vectors, `num_ws`
    to build W tensors, and `layers` for the bending UI and event
    validation. One snapshot channel serves all three instead of poking
    through to the model itself.
    """

    pkl_path: str
    z_dim: int
    num_ws: int
    layers: tuple[LayerInfo, ...] = ()


def _model_info(path: str, model: object) -> ModelInfo | None:
    """Build a `ModelInfo` from whatever the loader returned, or None.

    Duck-typed rather than an isinstance check: tests and future loaders
    stand in objects that are not `LoadedModel`. A double that omits the
    dimensions simply does not publish, never raises the loader thread. A
    `model.enumerate_layers()` that raises (a future wrapper generator that
    does not implement it faithfully, say) degrades to an empty catalog the
    same way `LoadedModel`'s own enumeration does, rather than failing the
    whole load.
    """
    z_dim = getattr(model, "z_dim", None)
    num_ws = getattr(model, "num_ws", None)
    if z_dim is None or num_ws is None:
        return None
    try:
        z_dim = int(z_dim)
        num_ws = int(num_ws)
    except (TypeError, ValueError):
        return None
    enumerate_layers = getattr(model, "enumerate_layers", None)
    try:
        layers = tuple(enumerate_layers()) if callable(enumerate_layers) else ()
    except Exception:
        logger.warning(
            "Could not enumerate layers for %s", path, exc_info=True
        )
        layers = ()
    return ModelInfo(pkl_path=str(path), z_dim=z_dim, num_ws=num_ws, layers=layers)


class LoadedModel:
    def __init__(self, pkl_path: str, G, device) -> None:
        self.pkl_path = pkl_path
        self.G = G
        self.device = device
        self._w_avg = G.mapping.w_avg
        self.z_dim = int(G.z_dim)
        self.num_ws = int(G.num_ws)
        self._c_dim = int(G.mapping.c_dim)
        self._applied_module_state: tuple | None = None
        self._named_modules: dict | None = None
        self._vec_fallback_logged = False
        self._keyframe_w_cache: dict = {}
        self._logged_once: set = set()
        self._log_cap_warned = False
        self._hook_key: tuple[str, ...] | None = None
        self._hook_handles: list = []
        self._manipulation = None
        self._manipulation_failed = False
        self._released = False
        # One instance per LoadedModel, never module-global (the legacy bug
        # this stage exists to shed): a device switch always builds a fresh
        # LoadedModel with its own fixed `device`, so a fresh SuperRes comes
        # along with it and loads/re-homes on its own first call instead of
        # inheriting a previous model's device.
        self._superres = SuperRes()
        # What the hooks act on for the frame being rendered right now, and
        # where the capture hook leaves what it grabbed. Empty between frames.
        self._frame_transforms: tuple[Transform, ...] = ()
        self._frame_capture = ""
        self._captured = None

    def _blended_w(self, latent_x, latent_y, truncation_psi):
        import torch

        corners = corner_seeds(latent_x, latent_y)
        unique_seeds = sorted({seed for seed, _ in corners})
        zs = np.zeros([len(unique_seeds), self.z_dim], dtype=np.float32)
        cs = np.zeros([len(unique_seeds), self._c_dim], dtype=np.float32)
        for index, seed in enumerate(unique_seeds):
            rnd = np.random.RandomState(seed)
            zs[index] = rnd.randn(self.z_dim)
            if self._c_dim > 0:
                cs[index, rnd.randint(self._c_dim)] = 1
        z_batch = torch.from_numpy(zs).to(self.device)
        c_batch = torch.from_numpy(cs).to(self.device)
        mapped = self.G.mapping(z=z_batch, c=c_batch, truncation_psi=truncation_psi)
        mapped = mapped - self._w_avg
        w_by_seed = dict(zip(unique_seeds, mapped))
        blended = torch.stack(
            [w_by_seed[seed] * weight for seed, weight in corners]
        ).sum(dim=0)
        return blended + self._w_avg

    def _log_vec_fallback(self) -> None:
        if self._vec_fallback_logged:
            return
        logger.warning(
            "Latent vector missing or wrong length for %s, using a deterministic fallback",
            self.pkl_path,
        )
        self._vec_fallback_logged = True

    def _z_for_vec(self, vec: tuple[float, ...]) -> np.ndarray:
        if len(vec) == self.z_dim:
            return np.asarray(vec, dtype=np.float32)
        self._log_vec_fallback()
        return np.random.RandomState(0).randn(self.z_dim).astype(np.float32)

    def _w_rows_for_vec(self, vec: tuple[float, ...], w_dim: int) -> np.ndarray:
        if len(vec) == 0 or len(vec) % w_dim != 0:
            self._log_vec_fallback()
            return np.random.RandomState(0).randn(1, w_dim).astype(np.float32)
        return np.asarray(vec, dtype=np.float32).reshape(-1, w_dim)

    def _vec_to_w(self, vec: tuple[float, ...], project: bool, truncation_psi: float):
        """A raw latent vector to a full `[num_ws, w_dim]` W tensor.

        `project=True` runs the mapping network (truncation applied there).
        `project=False` treats `vec` as W rows directly: too many rows
        truncate to `num_ws`, too few repeat the last row to fill it out.
        Old-app parity, and shared by both the standalone vector mode and
        `"vec"` keyframes.
        """
        import torch

        w_dim = self._w_avg.shape[-1]
        if project:
            z = self._z_for_vec(vec)
            z_batch = torch.from_numpy(z[None, :]).to(self.device)
            c_batch = torch.zeros(
                [1, self._c_dim], dtype=torch.float32, device=self.device
            )
            mapped = self.G.mapping(z=z_batch, c=c_batch, truncation_psi=truncation_psi)
            return mapped[0]
        rows = torch.from_numpy(self._w_rows_for_vec(vec, w_dim)).to(self.device)
        if rows.shape[0] >= self.num_ws:
            return rows[: self.num_ws]
        pad = rows[-1:].repeat(self.num_ws - rows.shape[0], 1)
        return torch.cat([rows, pad], dim=0)

    def _keyframe_to_w(self, keyframe: Keyframe, truncation_psi: float):
        if keyframe.kind == "vec":
            return self._vec_to_w(keyframe.vec, keyframe.project, truncation_psi)
        return self._blended_w(keyframe.seed_x, keyframe.seed_y, truncation_psi)

    def _cached_keyframe_w(self, keyframe: Keyframe, truncation_psi: float):
        """`_keyframe_to_w`, memoized so a static loop does not re-run the
        mapping network for both endpoints on every frame.

        Keyed by keyframe value plus truncation, sized 2 to 4 entries (a loop
        only ever needs its two active endpoints); a fresh `LoadedModel` per
        model switch invalidates it for free.
        """
        key = (keyframe, truncation_psi)
        cached = self._keyframe_w_cache.get(key)
        if cached is not None:
            return cached
        w = self._keyframe_to_w(keyframe, truncation_psi)
        self._keyframe_w_cache[key] = w
        if len(self._keyframe_w_cache) > _KEYFRAME_CACHE_SIZE:
            self._keyframe_w_cache.pop(next(iter(self._keyframe_w_cache)))
        return w

    def _loop_w(self, params: RenderParams):
        """Slerp between the previous and current keyframe.

        Negative indexing gives the closed-loop wrap for free: index 0
        interpolates in from the last keyframe, old-app parity.
        """
        keyframes = params.keyframes
        if not keyframes:
            return self._blended_w(0.0, 0.0, params.truncation_psi)
        index = params.loop_index
        w0 = self._cached_keyframe_w(keyframes[index - 1], params.truncation_psi)
        w1 = self._cached_keyframe_w(keyframes[index], params.truncation_psi)
        return slerp(params.loop_alpha, w0, w1)

    def _log_once(self, key, message: str, *args, exc_info: bool = False) -> None:
        """One warning per distinct cause, however many frames repeat it.

        The render thread hits these paths at frame rate, so a line per
        frame would bury the log and cost more than the failure itself. Past
        `_LOG_ONCE_CAP` distinct causes the model has bigger problems than a
        missing log line, and the set stops growing. The cap tripping is
        itself logged, once, so a session that goes quiet on new failure
        causes says why instead of just falling silent.
        """
        if key in self._logged_once:
            return
        if len(self._logged_once) >= _LOG_ONCE_CAP:
            if not self._log_cap_warned:
                self._log_cap_warned = True
                logger.warning(
                    "Reached %d distinct logged causes for %s, further "
                    "distinct causes will not be logged",
                    _LOG_ONCE_CAP,
                    self.pkl_path,
                )
            return
        self._logged_once.add(key)
        logger.warning(message, *args, exc_info=exc_info)

    def _synthesis_modules(self) -> dict:
        """Catalog name to submodule, resolved once for this loaded model.

        Names match `enumerate_layers`: relative to `G.synthesis`, with the
        synthesis module itself called "output". A `G` whose synthesis is
        not a module (a test double, a future wrapper) simply has no
        addressable layers rather than failing the frame.
        """
        if self._named_modules is None:
            named = getattr(getattr(self.G, "synthesis", None), "named_modules", None)
            if callable(named):
                self._named_modules = {
                    name or "output": module for name, module in named()
                }
            else:
                self._named_modules = {}
        return self._named_modules

    def _apply_module_state(self, params: RenderParams) -> None:
        """Push global noise, per-layer noise strength and per-layer ratios
        onto the layers that support them.

        Runs every frame, so it walks the network only when one of the three
        moved. These are Autolume's own additions to the stylegan2 synthesis
        layer, and every stylegan2 pkl arrives carrying them however it was
        trained, because the loader rebuilds it from that architecture
        (`legacy.create_networks`). A stylegan3-shaped pkl skips that rebuild
        and keeps its own pickled classes, which have none of the three, so
        the checks stay: we never invent an attribute a layer does not have. A
        layer that has dropped out of either sparse mapping is written back to
        neutral, so removing an override actually removes its effect.

        The ratio pair is swapped on its way onto the module, and only here.
        `SynthesisLayer.forward` binds `in_w = x.shape[-2]`, which is the
        activation's *height*, and then resizes to `(in_w * rx, in_h * ry)`,
        so the layer's slot 0 scales height and slot 1 scales width. State,
        presets, OSC and the panel all keep the x-then-y order their labels
        promise, and this one write is where the two orders meet. Measured on
        a real model rather than read off the source: without the swap, "Ratio
        x" of 2 on a 1024 model renders a 2048 by 1024 frame.
        """
        state = (params.global_noise, params.layer_noise, params.layer_ratios)
        if state == self._applied_module_state:
            return
        for name, module in self._synthesis_modules().items():
            if hasattr(module, "global_noise"):
                module.global_noise = params.global_noise
            if hasattr(module, "noise_regulator"):
                module.noise_regulator = params.layer_noise.get(name, 0.0)
            if hasattr(module, "ratio"):
                rx, ry = params.layer_ratios.get(name, (1.0, 1.0))
                module.ratio = (ry, rx)
        # Copied, never aliased: the snapshot these came from is shared with
        # the control thread and must not be reachable from here.
        self._applied_module_state = (
            params.global_noise,
            dict(params.layer_noise),
            dict(params.layer_ratios),
        )

    def _ensure_manipulation(self) -> None:
        """The operator library, loaded the first time a transform needs it.

        Imported here rather than at module scope so a session that never
        bends anything never pays for kornia. A failed import is remembered
        as a failure, not just as "not yet succeeded": this runs from
        `_sync_hooks`, which now fires on every frame that has a transform
        (moved there to fix an unrelated bug where a transform on an
        already-captured layer never applied), and a broken install would
        otherwise retry the full import machinery, including kornia's own
        slow import chain, at frame rate. This model is never given a
        working bending install mid-session, so the failure does not need
        to be retryable; a fresh `LoadedModel` on the next model load is the
        natural place a fixed install gets picked up again.
        """
        if self._manipulation is not None or self._manipulation_failed:
            return
        try:
            from autolume.bending.transform_layers import ManipulationLayer

            self._manipulation = ManipulationLayer()
        except Exception:
            self._manipulation_failed = True
            self._log_once(
                ("manipulation",),
                "Could not load the bending operators, transforms stay inactive",
                exc_info=True,
            )

    def release(self) -> None:
        """Drop this model's forward hooks so `G` is freed by refcount.

        The hooks form a cycle, `G -> module._forward_hooks -> closure ->
        LoadedModel -> G`, that would otherwise keep this retired model's
        VRAM alive until a generational GC pass runs instead of the instant
        the last reference to it drops. `ModelHost` calls this the moment a
        model stops being `current()`. This runs after the incoming model
        has already been allocated, so it does nothing for that
        allocation's peak; what it buys is not holding this model's memory
        any longer than the runtime needs to, so it is already gone well
        before the *next* reload needs the room.

        `_released` also gates `_sync_hooks`: without it, a frame still in
        flight on this retired model would re-register hooks nobody holds
        the handles for, recreating the exact cycle this method exists to
        break. Idempotent either way.
        """
        if self._released:
            return
        self._released = True
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
        self._hook_key = None
        self._manipulation = None

    def _sync_hooks(self, params: RenderParams) -> None:
        """Hold one forward hook on every layer this frame reads or edits.

        Registration follows the layer name key and nothing else, so turning
        a transform's parameters, or the whole chain's values, costs no hook
        traffic at all. An empty key means the network carries no hooks.
        """
        if self._released:
            return
        # Loaded before the key compare, not after: a capture layer and a
        # transform on that same layer fold into one key, so the first
        # transform of a session can arrive without the key moving at all.
        if params.transforms:
            self._ensure_manipulation()
        key = hook_layer_names(params)
        if key == self._hook_key:
            return
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
        self._hook_key = key
        modules = self._synthesis_modules()
        for name in key:
            module = modules.get(name)
            if module is None:
                self._log_once(
                    ("layer", name),
                    "Layer %s is not part of %s, skipping it",
                    name,
                    self.pkl_path,
                )
                continue
            register = getattr(module, "register_forward_hook", None)
            if callable(register):
                self._hook_handles.append(register(self._bend_hook(name)))

    def _bend_hook(self, name: str):
        def _hook(_module, _inputs, output):
            if isinstance(output, tuple):
                tensor = output[0] if output else None
            else:
                tensor = output
            if getattr(tensor, "ndim", 0) not in (4, 5):
                return None
            bent, applied = self._apply_transforms(name, tensor)
            if name == self._frame_capture:
                # A 5D activation is a G-CNN group layout (N, C, G, H, W) and
                # the image comes from the group mean, which is also the shape
                # the layer catalog reports.
                self._captured = bent.mean(2) if bent.ndim == 5 else bent
            if not applied:
                return None
            if isinstance(output, tuple):
                return (bent,) + tuple(output[1:])
            return bent

        return _hook

    def _apply_transforms(self, name: str, tensor):
        """Every transform aimed at `name`, in chain order.

        Called from a forward hook, so it cannot raise a Python exception: a
        transform that fails is logged once for that cause and dropped, and
        the frame renders with whatever the rest of the chain produced.
        """
        manipulation = self._manipulation
        if manipulation is None:
            return tensor, False
        channels = int(tensor.shape[1])
        applied = False
        for transform in self._frame_transforms:
            if transform.layer != name:
                continue
            indices = usable_indices(transform, channels)
            if len(indices) != len(transform.indices):
                self._log_once(
                    ("indices", transform.op, name, channels),
                    "Bending %s on %s selects channels this layer does not have, "
                    "it only has %d",
                    transform.op,
                    name,
                    channels,
                )
            if not indices:
                continue
            try:
                tensor = manipulation(tensor, manipulation_dict(transform, indices))
                applied = True
            except Exception as exc:
                self._log_once(
                    ("transform", transform.op, name, str(exc)),
                    "Bending %s on %s failed, skipping it: %s",
                    transform.op,
                    name,
                    exc,
                )
        return tensor, applied

    def _direction(self, params: RenderParams):
        import torch

        width = int(self._w_avg.shape[-1])
        delta, mismatched = direction_delta(params, width)
        for index in mismatched:
            self._log_once(
                ("direction", index, len(params.directions[index])),
                "Adjuster direction %d has %d values but W is %d wide here, skipping it",
                index + 1,
                len(params.directions[index]),
                width,
            )
        if delta is None:
            return None
        return torch.from_numpy(delta).to(self.device)

    def enumerate_layers(self) -> tuple[LayerInfo, ...]:
        """Build the layer catalog with one hooked dry synthesis pass.

        Hooks every synthesis submodule by its `named_modules()` name so a
        later transform hook can look a module up by the same name, then
        synthesizes once with a deterministic fallback W (the same
        RandomState(0) technique `_w_rows_for_vec` falls back to, but
        without its warning, since this runs on every load). Forward hooks
        fire on a module only after its own forward() returns, which is
        after all its children have already fired theirs, so the top-level
        synthesis module (named '' by `named_modules()`, recorded here as
        "output") is always last. Only 4D/5D outputs count as layers. A 5D
        output is a G-CNN group dimension laid out `(N, C, G, H, W)`, so
        `shape[1]` is the channel count there too, not `shape[-3]` (the
        group count). Hooks are removed in `finally` even if the dry pass
        raises: an exotic architecture must not leave hooks on a model
        about to render, and must not block the load either, so any
        failure here logs one line and yields an empty catalog instead of
        raising.
        """
        import torch

        layers: list[LayerInfo] = []
        handles = []

        def _catalog_hook(name: str):
            def _hook(_module, _inputs, output):
                if isinstance(output, tuple):
                    output = output[0] if output else None
                shape = getattr(output, "shape", None)
                if shape is None or len(shape) not in (4, 5):
                    return
                layers.append(
                    LayerInfo(
                        name=name or "output",
                        channels=int(shape[1]),
                        width=int(shape[-1]),
                        height=int(shape[-2]),
                    )
                )

            return _hook

        try:
            synthesis = self.G.synthesis
            for name, module in synthesis.named_modules():
                handles.append(module.register_forward_hook(_catalog_hook(name)))
            w_dim = self._w_avg.shape[-1]
            row = np.random.RandomState(0).randn(1, w_dim).astype(np.float32)
            ws = torch.from_numpy(row).to(self.device).repeat(self.num_ws, 1)
            with torch.no_grad():
                synthesis(ws.unsqueeze(0), noise_mode="const")
        except Exception:
            logger.warning(
                "Could not enumerate layers for %s", self.pkl_path, exc_info=True
            )
            return ()
        finally:
            for handle in handles:
                handle.remove()
        return tuple(layers)

    def render_frame(self, params: RenderParams, frame_index: int) -> np.ndarray:
        import torch

        with torch.no_grad():
            self._apply_module_state(params)
            self._sync_hooks(params)
            if params.mode == "vec":
                ws = self._vec_to_w(
                    params.latent_vec, params.latent_project, params.truncation_psi
                )
            elif params.mode == "loop":
                ws = self._loop_w(params)
            else:
                ws = self._blended_w(
                    params.latent_x, params.latent_y, params.truncation_psi
                )
            direction = self._direction(params)
            if direction is not None:
                ws = ws + direction
            mode = noise_mode(params)
            # Only "random" draws from torch's global generator, and seeding it
            # is a process wide side effect, so the other modes leave it alone.
            if mode == "random":
                torch.manual_seed(effective_noise_seed(params, frame_index))
            self._frame_transforms = params.transforms
            self._frame_capture = params.capture_layer
            self._captured = None
            # force_fp32 is legacy, CUDA-only semantics: passed through only
            # when set, so a fake synthesis in a test that only accepts
            # (ws, noise_mode) keeps working, and a real network's own
            # default (False) governs whenever it is not.
            synthesis_kwargs = {"noise_mode": mode}
            if params.force_fp32:
                synthesis_kwargs["force_fp32"] = True
            try:
                output = self.G.synthesis(ws.unsqueeze(0), **synthesis_kwargs)
            finally:
                captured = self._captured
                self._frame_transforms = ()
                self._frame_capture = ""
                self._captured = None
            # Autolume's custom stylegan2 synthesis returns (img, rgb_list);
            # standard stylegan synthesis returns the img tensor directly.
            if isinstance(output, tuple):
                output = output[0]
            activation = output if captured is None else captured
            image = derive_float_image(activation[0], params)
            if params.use_superres:
                image = self._superres.apply(image, self.device)
            return to_uint8_frame(image)


def load_model(path: str, device=None) -> LoadedModel:
    import torch

    import dnnlib
    from torch_utils import legacy

    device = device or pick_device()
    with dnnlib.util.open_url(str(path), verbose=False) as f:
        data = legacy.load_network_pkl(f, custom=True)
    G = data["G_ema"].eval().requires_grad_(False).to(device)
    logger.info("Loaded %s on %s", path, device)
    return LoadedModel(str(path), G, device)


def load_discriminator(path: str):
    """The discriminator out of a network pkl.

    A merged model is written with the discriminator of the model it was
    mixed with, matching the offline mixing tool, because
    `legacy.load_network_pkl` asserts all three of G, D and G_ema are
    modules and a file written without one could never be loaded back.
    """
    import dnnlib
    from torch_utils import legacy

    with dnnlib.util.open_url(str(path), verbose=False) as f:
        data = legacy.load_network_pkl(f, custom=True)
    return data["D"]


def _release_quietly(model: object) -> None:
    """Call `model.release()` if it has one, never raising into the caller.

    Duck-typed and best-effort: a loader's test double is rarely a real
    `LoadedModel` and need not implement `release`, and a real one failing to
    remove a handle must not take the loader thread down over a model that is
    being thrown away anyway.
    """
    release = getattr(model, "release", None)
    if not callable(release):
        return
    try:
        release()
    except Exception:
        logger.exception("Failed releasing a retired model")


class ModelHost:
    """Owns the loaded model and a background loader thread.

    request_load never blocks the caller. Concurrent requests coalesce to
    the newest path. The render thread reads current() under the lock.
    """

    def __init__(self, loader: Callable[[str], LoadedModel] | None = None) -> None:
        self._loader = loader or load_model
        self._lock = threading.Lock()
        self._current: LoadedModel | None = None
        self._error: str | None = None
        self._pending: str | None = None
        # None means "this load, whatever it is, uses whatever device is
        # currently selected" (see `_device_name` below); the loader thread
        # decides at load time, not here. Set to a name only while a
        # request_device-triggered reload is actually in flight.
        self._pending_device: str | None = None
        # The device every load, not only a switch, resolves against:
        # "auto" reproduces the original pre-device-switching behavior
        # (the loader picks, one argument, no explicit device), so a plain
        # `request_load` after a device has been chosen does not silently
        # abandon it and go back to `pick_device()`.
        self._device_name: str = "auto"
        # Bounds `_load_default`'s device-resolution retry to one extra
        # attempt per path: the restored `_device_name` is expected to
        # always resolve, but that is an assumption about `.device`, not an
        # enforced property, and a model reporting a device string
        # `resolve_device` never matches would otherwise retry forever.
        self._retry_attempted_path: str | None = None
        # Slot B and the mix built from the pair. `_current` stays model A
        # throughout: only `current()` chooses between it and `_mixed`, so
        # every device and load path above is untouched by mixing.
        self._current_b: LoadedModel | None = None
        self._pending_b: str | None = None
        self._mixed: LoadedModel | None = None
        self._mixing_enabled = False
        self._combined_layers: tuple[str, ...] = ()
        self._pending_mix: tuple[str, ...] | None = None
        self._pending_save: str | None = None
        # The catalogs for A and for the mix are kept apart because
        # `info_store` publishes whichever one `current()` is rendering, and
        # a mix that truncates depth has different layers from its own
        # source A.
        self._info_a: ModelInfo | None = None
        self._info_mixed: ModelInfo | None = None
        self.info_store: LatestValueStore[ModelInfo | None] = LatestValueStore(None)
        self.device_store: LatestValueStore[DeviceStatus] = LatestValueStore(
            DeviceStatus()
        )
        self.mix_save_store: LatestValueStore[MixSaveStatus] = LatestValueStore(
            MixSaveStatus()
        )
        self._wakeup = threading.Event()
        self._running = True
        self._thread = threading.Thread(
            target=self._run, name="model-loader", daemon=True
        )
        self._thread.start()

    def request_load(self, path: str) -> None:
        with self._lock:
            self._pending = str(path)
            self._pending_device = None
            # A fresh, explicit load intent gets its own bounded retry
            # budget, regardless of whatever an earlier, unrelated cycle
            # (for this path or a different one, abandoned mid-retry by a
            # request that superseded it) left behind.
            self._retry_attempted_path = None
        self._wakeup.set()

    def request_device(self, name: str) -> None:
        """Point the host at `name` for this and every future load.

        Never clobbers a pkl request already in flight: if one is loading,
        the device applies to that same incoming model instead of reloading
        whatever is already `current()`, which would otherwise strand the
        pkl on its way in behind a model nobody asked for anymore. If
        nothing is in flight, this reloads whatever is currently loaded. If
        nothing is loaded and nothing is loading, `name` is only
        remembered: the very first `request_load` resolves and uses it, so
        a device picked before any model exists is never silently dropped.
        `name` is resolved and validated on the loader thread, never here,
        so this never blocks and never raises.
        """
        with self._lock:
            self._device_name = str(name)
            if self._pending is not None:
                self._pending_device = self._device_name
                self._retry_attempted_path = None
            elif self._current is not None:
                self._pending = self._current.pkl_path
                self._pending_device = self._device_name
                self._retry_attempted_path = None
            else:
                return
        self._wakeup.set()

    def request_load_b(self, path: str) -> None:
        """Load the second mixing source into slot B.

        Independent of slot A: the render loop never sees slot B, so a load
        here never interrupts what is on screen.
        """
        with self._lock:
            self._pending_b = str(path)
        self._wakeup.set()

    def request_mix(self, combined_layers) -> None:
        """Assemble the mixed network on the loader thread.

        `combined_layers` is one entry per name in `mixing.conv_names` of
        the loaded pair, padded to the longer of the two. Never blocks and
        never raises: a selection that no longer lines up with the models
        under it, a pair that cannot be assembled, or a slot that is empty
        all report through `error()` and leave model A rendering.
        """
        # A bare string iterates one character at a time and would be read as
        # a selection rather than rejected, so it is ruled out by name.
        if isinstance(combined_layers, str):
            logger.warning("Ignoring a mix request that is not a sequence of entries")
            return
        try:
            entries = tuple(str(entry) for entry in combined_layers)
        except Exception:
            logger.warning("Ignoring a mix request that is not a sequence of entries")
            return
        with self._lock:
            self._combined_layers = entries
            self._pending_mix = entries
        self._wakeup.set()

    def set_mixing_enabled(self, enabled: bool) -> None:
        """Choose between the mixed network and model A as what renders.

        Never discards a built mix when it turns off: flipping between the
        pair and the mix is a performance gesture, and rebuilding a whole
        generator every time would cost seconds. A build is queued only
        when mixing turns on with a selection and nothing built.
        """
        enabled = bool(enabled)
        with self._lock:
            if enabled == self._mixing_enabled:
                return
            self._mixing_enabled = enabled
            queue = enabled and self._mixed is None and bool(self._combined_layers)
            if queue:
                self._pending_mix = self._combined_layers
        self._publish_info()
        if queue:
            self._wakeup.set()

    def request_save_mix(self, output_name: str) -> None:
        """Write the current selection to `<output_name>.pkl` in the models
        folder. The outcome lands on `mix_save_store`.
        """
        with self._lock:
            self._pending_save = str(output_name)
        self._wakeup.set()

    def current(self) -> LoadedModel | None:
        """What the render loop draws: the mix while it is on, else model A."""
        with self._lock:
            if self._mixing_enabled and self._mixed is not None:
                return self._mixed
            return self._current

    def current_a(self) -> LoadedModel | None:
        """Slot A itself, never the mix built from it.

        `current()` deliberately answers "what is on screen", which is the mix
        whenever one is built and mixing is on. A caller that needs model A as a
        *mixing source* cannot use it: the mixing panel derives its rows,
        its selection length and its cascades from model A's own layer names,
        and reading them off the mix gives the names of the network the
        selection produced rather than the one it applies to.

        Gating such a read on `mixing_enabled()` instead does not work and is
        the bug this exists to close: `_retire_mix_locked` clears `_mixed` but
        leaves `_mixing_enabled` set, so a model A swap while mixing is on
        leaves the gate shut and the caller holding names from a model that is
        no longer loaded.
        """
        with self._lock:
            return self._current

    def current_b(self) -> LoadedModel | None:
        with self._lock:
            return self._current_b

    def pending_b(self) -> str | None:
        with self._lock:
            return self._pending_b

    def mixing_enabled(self) -> bool:
        with self._lock:
            return self._mixing_enabled

    def error(self) -> str | None:
        with self._lock:
            return self._error

    def pending(self) -> str | None:
        """The model being loaded right now, or None if none is.

        Both facts from one read, because the UI wants to name what is loading
        and two reads could straddle a load finishing and report a state this
        host was never in.
        """
        with self._lock:
            return self._pending

    def loading(self) -> bool:
        with self._lock:
            return self._pending is not None

    def stop(self) -> None:
        self._running = False
        self._wakeup.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while self._running:
            self._wakeup.wait()
            self._wakeup.clear()
            if not self._running:
                return
            with self._lock:
                path = self._pending
                pending_device = self._pending_device
                sticky_device = self._device_name
                path_b = self._pending_b
                entries = self._pending_mix
                save_name = self._pending_save
            # Slot A first, so a mix queued in the same wakeup assembles
            # from the pair the requests actually meant.
            if path is not None:
                if pending_device is None:
                    self._load_default(path, sticky_device)
                else:
                    self._load_on_device(path, pending_device)
            if path_b is not None:
                self._load_b(path_b)
            if entries is not None:
                self._build_mix(entries)
            if save_name is not None:
                self._save_mix(save_name)
            if self._has_work():
                self._wakeup.set()

    def _has_work(self) -> bool:
        with self._lock:
            return (
                self._pending is not None
                or self._pending_b is not None
                or self._pending_mix is not None
                or self._pending_save is not None
            )

    def _publish_info(self) -> None:
        """Publish the catalog of whatever `current()` now returns."""
        with self._lock:
            active = (
                self._info_mixed
                if self._mixing_enabled and self._mixed is not None
                else self._info_a
            )
        self.info_store.set(active)

    def _retire_mix_locked(self) -> LoadedModel | None:
        """Drop the built mix and queue a rebuild if one is still wanted.

        Called with the lock held whenever either source changes, a device
        switch included. The mixed generator holds copies of the sources'
        weights and sits on the device slot A had when it was built, so it
        can never outlive the pair it came from. Returns the retired model
        for the caller to release outside the lock.
        """
        stale = self._mixed
        self._mixed = None
        self._info_mixed = None
        if self._mixing_enabled and self._combined_layers:
            self._pending_mix = self._combined_layers
        return stale

    def _drop_mix(self, error: str | None) -> None:
        """Stop rendering the mix and, when there is one, report why."""
        with self._lock:
            stale = self._mixed
            self._mixed = None
            self._info_mixed = None
            if error is not None:
                self._error = error
        _release_quietly(stale)
        self._publish_info()

    def _load_b(self, path: str) -> None:
        """Load the second mixing source, always onto the CPU.

        Slot B is never rendered. It exists only as a bag of weights for
        `combine`, which copies what it needs into a generator of its own,
        and `load_state_dict` copies across devices for free. Keeping B off
        the render device costs nothing, halves the VRAM a mixing session
        needs, and leaves it untouched by slot A's device switching.
        """
        import torch

        try:
            model = self._loader(path, device=torch.device("cpu"))
        except Exception as exc:
            logger.exception("Failed to load the mixing model %s", path)
            with self._lock:
                won = self._pending_b == path
                if won:
                    self._pending_b = None
                    self._error = str(exc)
            return
        with self._lock:
            won = self._pending_b == path
            if won:
                previous = self._current_b
                self._current_b = model
                self._pending_b = None
                # Deliberately not clearing `_error`: it belongs to whatever
                # last failed, and slot B succeeding says nothing about a
                # slot A that is still not loaded.
                stale_mix = self._retire_mix_locked()
        if won:
            _release_quietly(previous)
            _release_quietly(stale_mix)
            self._publish_info()
        else:
            _release_quietly(model)

    def _build_mix(self, entries: tuple[str, ...]) -> None:
        """Assemble `entries` into the network the render loop draws.

        Both sources stay loaded and untouched: the mix is a third
        generator holding copies. Every failure leaves model A rendering,
        so there is never a black frame, only a status line.
        """
        with self._lock:
            if self._pending_mix == entries:
                self._pending_mix = None
            model_a = self._current
            model_b = self._current_b
            enabled = self._mixing_enabled
        if not entries:
            self._drop_mix(None)
            return
        if not enabled:
            return
        if model_a is None or model_b is None:
            self._drop_mix("Load a model in both slots to mix them.")
            return
        try:
            network = combine(model_a.G, model_b.G, entries)
            network = network.eval().requires_grad_(False).to(model_a.device)
            mixed = LoadedModel(model_a.pkl_path, network, model_a.device)
        except Exception as exc:
            logger.warning("Could not build the mixed model: %s", exc)
            self._drop_mix(str(exc))
            return
        # Enumerated before publishing, exactly as a plain load is: the dry
        # synthesis pass hooks this model's submodules and must be finished
        # before the render thread can reach it through current().
        info = _model_info(model_a.pkl_path, mixed)
        with self._lock:
            won = (
                self._current is model_a
                and self._current_b is model_b
                and self._pending_mix is None
            )
            if won:
                previous = self._mixed
                self._mixed = mixed
                self._info_mixed = info
                self._error = None
        if not won:
            # A source moved, or a newer selection arrived, while this was
            # building: it was assembled from a pair that is no longer the
            # pair, so it is dropped rather than shown.
            _release_quietly(mixed)
            return
        _release_quietly(previous)
        self._publish_info()

    def _save_mix(self, output_name: str) -> None:
        """Write the current selection to `<output_name>.pkl`.

        Assembled fresh on the CPU rather than pickling the network on
        screen: the file stays loadable on a machine with a different
        device, and the rendering mix is never touched. The name is reduced
        to a bare file name so a typed path can never write outside the
        models folder.
        """
        import pickle

        from utils.model_dir import ensure_models_dir

        with self._lock:
            self._pending_save = None
            model_a = self._current
            model_b = self._current_b
            entries = self._combined_layers
        name = os.path.basename(str(output_name)).strip()
        if name.lower().endswith(".pkl"):
            name = name[: -len(".pkl")]
        try:
            if not name:
                raise ValueError("Give the merged model a file name.")
            if model_a is None or model_b is None:
                raise ValueError("Load a model in both slots to mix them.")
            merged = combine(model_a.G, model_b.G, entries)
            merged = merged.eval().requires_grad_(False)
            data = {
                "G": merged,
                "G_ema": merged,
                "D": load_discriminator(model_b.pkl_path),
            }
            path = str(os.path.join(ensure_models_dir(), f"{name}.pkl"))
            with open(path, "wb") as handle:
                pickle.dump(data, handle)
        except Exception as exc:
            logger.exception("Failed to save the merged model")
            self.mix_save_store.set(MixSaveStatus(error=str(exc)))
            return
        logger.info("Saved the merged model to %s", path)
        self.mix_save_store.set(MixSaveStatus(path=path))

    def _load_default(self, path: str, device_name: str) -> None:
        """The ordinary pkl load, resolved against `device_name` (the
        host's current device selection, "auto" unless request_device has
        ever been called) rather than whatever the loader itself would pick.

        "auto" still calls the loader with a single argument, exactly as
        before device switching existed, so a loader test double taking
        just `path` keeps working; any other device name requires a loader
        that accepts `device=`. Resolution failure (a device remembered
        from an earlier switch that no longer resolves) reports through
        `device_store` only, the same channel a switch failure uses, and
        retries `path` once with the device name restored to whatever is
        actually running: not a terminal, `error()`-and-`info_store`-
        clearing failure on that first attempt, since the retry is
        expected to succeed. A path already sitting in `_current` short
        circuits instead of retrying at all, matching `_load_on_device`'s
        own `already_current` case: there is nothing to reload. If the
        restored name fails to resolve too, this stops rather than
        retrying again — `.device` always resolving is an assumption, not
        an enforced property, and one this method does not trust past a
        single retry. Any other load failure (a broken pkl, say) still
        reports through the plain `error()`/`info_store` channel, since
        the device was never at fault there.
        """
        try:
            device = None if device_name == "auto" else resolve_device(device_name)
        except DeviceUnavailable as exc:
            logger.warning("Could not load %s on device %s: %s", path, device_name, exc)
            gave_up = False
            with self._lock:
                won = self._pending == path and self._pending_device is None
                active = getattr(self._current, "device", None)
                already_current = (
                    self._current is not None and self._current.pkl_path == path
                )
                if won:
                    # Never keep a device name that just failed to resolve:
                    # fall back to whatever is actually running (or "auto"
                    # if nothing is), so this host's own state stays
                    # self-consistent no matter whether anything
                    # downstream ever notices and re-requests a good
                    # value.
                    self._device_name = str(active) if active is not None else "auto"
                    if already_current:
                        # `path` is already loaded and rendering fine, the
                        # same as `_load_on_device`'s own already_current
                        # case: only the sticky selection was bad, nothing
                        # to retry.
                        self._pending = None
                        self._retry_attempted_path = None
                    elif self._retry_attempted_path == path:
                        # The retry itself failed too, on a name that was
                        # supposed to always resolve. Stop rather than
                        # spin: a `.device` this never anticipated
                        # degrades to one reported failure, not a hot
                        # loop taking the lock and re-waking every
                        # iteration.
                        gave_up = True
                        self._pending = None
                        self._error = str(exc)
                        self._retry_attempted_path = None
                        self._info_a = None
                    else:
                        # First failure for this path: `_pending` stays
                        # put, so `_run`'s own re-wake retries it through
                        # this same method next, now with the device name
                        # just restored above.
                        self._retry_attempted_path = path
            if won:
                self.device_store.set(
                    DeviceStatus(
                        active=str(active) if active is not None else None,
                        requested=device_name,
                        error=str(exc),
                    )
                )
                if gave_up:
                    self._publish_info()
            return
        try:
            model = (
                self._loader(path)
                if device is None
                else self._loader(path, device=device)
            )
            # Enumerated before publishing: the layer catalog's dry
            # synthesis pass attaches and removes hooks on this model's
            # submodules, and that must be finished before the render
            # thread can ever see this model through current().
            info = _model_info(path, model)
            with self._lock:
                won = self._pending == path and self._pending_device is None
                if won:
                    previous = self._current
                    self._current = model
                    self._error = None
                    self._pending = None
                    self._retry_attempted_path = None
                    self._info_a = info
                    stale_mix = self._retire_mix_locked()
                # else: a newer request arrived while loading; loop again
            if won:
                _release_quietly(previous)
                _release_quietly(stale_mix)
                self._publish_info()
                self.device_store.set(
                    DeviceStatus(
                        active=str(getattr(model, "device", None))
                        if getattr(model, "device", None) is not None
                        else None,
                        requested=device_name,
                        error=None,
                    )
                )
        except Exception as exc:
            logger.exception("Failed to load model %s", path)
            with self._lock:
                won = self._pending == path and self._pending_device is None
                if won:
                    self._error = str(exc)
                    self._pending = None
                    self._retry_attempted_path = None
                    self._info_a = None
            if won:
                self._publish_info()

    def _load_on_device(self, path: str, device_name: str) -> None:
        """Reload `path` onto `device_name`, publishing the outcome.

        Never touches `_current` on failure, whether the device name cannot
        be resolved or the reload itself fails: the render loop keeps
        showing the previous model on its previous device, and the control
        loop reverts the registry value once it sees `device_store`'s error.
        """
        try:
            device = resolve_device(device_name)
            model = self._loader(path, device=device)
        except Exception as exc:
            logger.warning("Could not switch to device %s: %s", device_name, exc)
            with self._lock:
                won = (
                    self._pending == path and self._pending_device == device_name
                )
                active = getattr(self._current, "device", None)
                already_current = (
                    self._current is not None and self._current.pkl_path == path
                )
                if won:
                    # Never keep a device name that just failed: fall back
                    # to whatever is actually running (or "auto" if
                    # nothing is), the same value the status below
                    # publishes, so this host's own state stays self
                    # consistent regardless of what the control loop does
                    # next (see the Critical this fixes).
                    self._device_name = str(active) if active is not None else "auto"
                    if already_current:
                        # `path` is the model already settled in
                        # `_current`; there is nothing to retry, the
                        # render loop is already showing it, just on its
                        # previous device.
                        self._pending = None
                        self._pending_device = None
                    else:
                        # `path` was never actually loaded: this reload
                        # was redirecting a pkl request already in flight
                        # (`request_device` on a load that has not landed
                        # yet). Dropping it here would silently discard a
                        # model the user is still waiting on, so `_pending`
                        # stays put and only `_pending_device` clears, so
                        # `_run`'s own re-wake retries it as a plain load
                        # on the device just restored above.
                        self._pending_device = None
            if won:
                self.device_store.set(
                    DeviceStatus(
                        active=str(active) if active is not None else None,
                        requested=device_name,
                        error=str(exc),
                    )
                )
            return
        info = _model_info(path, model)
        with self._lock:
            won = self._pending == path and self._pending_device == device_name
            if won:
                previous = self._current
                self._current = model
                self._error = None
                self._pending = None
                self._pending_device = None
                # Hygiene, not correctness: this method never reads the
                # marker itself, but a stale one left over from an
                # earlier _load_default retry cycle for this same path
                # must not survive to confuse a later one.
                self._retry_attempted_path = None
                self._info_a = info
                stale_mix = self._retire_mix_locked()
        if won:
            _release_quietly(previous)
            _release_quietly(stale_mix)
            self._publish_info()
            self.device_store.set(
                DeviceStatus(
                    active=str(getattr(model, "device", device)),
                    requested=device_name,
                    error=None,
                )
            )
