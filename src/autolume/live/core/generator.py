"""Model loading, latent navigation, and synthesis.

All torch-and-model specifics live here so the render loop stays
orchestration-only. Latent navigation is the seed-grid bilinear walk:
every integer point of seed space owns a deterministic z draw, and a
continuous position blends the four surrounding seeds in w space.
Ported from balagan (latent_navigator.py).
"""

import logging
import math
import threading
from dataclasses import dataclass
from typing import Callable

import numpy as np

from autolume.live.core.params import Keyframe, RenderParams
from autolume.live.core.store import LatestValueStore

logger = logging.getLogger(__name__)

_SEED_MASK = (1 << 32) - 1
_BILINEAR_CORNERS = ((0, 0), (1, 0), (0, 1), (1, 1))
_SLERP_COLINEAR_THRESHOLD = 0.9995
_KEYFRAME_CACHE_SIZE = 4


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


def pick_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
        self._applied_global_noise: float | None = None
        self._vec_fallback_logged = False
        self._keyframe_w_cache: dict = {}

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

    def _apply_global_noise(self, value: float) -> None:
        """Push the global noise scale onto the layers that support it.

        Runs every frame, so it walks the network only when the value moved.
        Autolume's custom architecture defines `global_noise` on its noise
        layers, stock StyleGAN networks do not, and we never invent it.
        """
        if value == self._applied_global_noise:
            return
        for module in self.G.modules():
            if hasattr(module, "global_noise"):
                module.global_noise = value
        self._applied_global_noise = value

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

        def _make_hook(name: str):
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
                handles.append(module.register_forward_hook(_make_hook(name)))
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
            self._apply_global_noise(params.global_noise)
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
            mode = noise_mode(params)
            # Only "random" draws from torch's global generator, and seeding it
            # is a process wide side effect, so the other modes leave it alone.
            if mode == "random":
                torch.manual_seed(effective_noise_seed(params, frame_index))
            output = self.G.synthesis(ws.unsqueeze(0), noise_mode=mode)
            # Autolume's custom stylegan2 synthesis returns (img, rgb_list);
            # standard stylegan synthesis returns the img tensor directly.
            if isinstance(output, tuple):
                output = output[0]
            image = (output[0] * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            return image.permute(1, 2, 0).contiguous().cpu().numpy()


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
        self.info_store: LatestValueStore[ModelInfo | None] = LatestValueStore(None)
        self._wakeup = threading.Event()
        self._running = True
        self._thread = threading.Thread(
            target=self._run, name="model-loader", daemon=True
        )
        self._thread.start()

    def request_load(self, path: str) -> None:
        with self._lock:
            self._pending = str(path)
        self._wakeup.set()

    def current(self) -> LoadedModel | None:
        with self._lock:
            return self._current

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
            if path is None:
                continue
            try:
                model = self._loader(path)
                # Enumerated before publishing: the layer catalog's dry
                # synthesis pass attaches and removes hooks on this model's
                # submodules, and that must be finished before the render
                # thread can ever see this model through current().
                info = _model_info(path, model)
                with self._lock:
                    won = self._pending == path
                    if won:
                        self._current = model
                        self._error = None
                        self._pending = None
                    # else: a newer request arrived while loading; loop again
                if won:
                    self.info_store.set(info)
            except Exception as exc:
                logger.exception("Failed to load model %s", path)
                with self._lock:
                    won = self._pending == path
                    if won:
                        self._error = str(exc)
                        self._pending = None
                if won:
                    self.info_store.set(None)
            if self.loading():
                self._wakeup.set()
