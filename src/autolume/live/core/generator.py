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
from typing import Callable

import numpy as np

from autolume.live.core.params import RenderParams

logger = logging.getLogger(__name__)

_SEED_MASK = (1 << 32) - 1
_BILINEAR_CORNERS = ((0, 0), (1, 0), (0, 1), (1, 1))


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


class LoadedModel:
    def __init__(self, pkl_path: str, G, device) -> None:
        self.pkl_path = pkl_path
        self.G = G
        self.device = device
        self._w_avg = G.mapping.w_avg
        self._z_dim = int(G.z_dim)
        self._c_dim = int(G.mapping.c_dim)
        self._applied_global_noise: float | None = None

    def _blended_w(self, latent_x, latent_y, truncation_psi):
        import torch

        corners = corner_seeds(latent_x, latent_y)
        unique_seeds = sorted({seed for seed, _ in corners})
        zs = np.zeros([len(unique_seeds), self._z_dim], dtype=np.float32)
        cs = np.zeros([len(unique_seeds), self._c_dim], dtype=np.float32)
        for index, seed in enumerate(unique_seeds):
            rnd = np.random.RandomState(seed)
            zs[index] = rnd.randn(self._z_dim)
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

    def render_frame(self, params: RenderParams, frame_index: int) -> np.ndarray:
        import torch

        with torch.no_grad():
            self._apply_global_noise(params.global_noise)
            ws = self._blended_w(
                params.latent_x, params.latent_y, params.truncation_psi
            )
            torch.manual_seed(effective_noise_seed(params, frame_index))
            output = self.G.synthesis(ws.unsqueeze(0), noise_mode=noise_mode(params))
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
                with self._lock:
                    if self._pending == path:
                        self._current = model
                        self._error = None
                        self._pending = None
                    # else: a newer request arrived while loading; loop again
            except Exception as exc:
                logger.exception("Failed to load model %s", path)
                with self._lock:
                    if self._pending == path:
                        self._error = str(exc)
                        self._pending = None
            if self.loading():
                self._wakeup.set()
