import multiprocessing
import os
import sys

IS_FROZEN = getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")

if sys.platform == 'darwin':
    # Must be set before torch is imported.
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

if sys.platform == 'linux' and os.environ.get('WAYLAND_DISPLAY'):
    # Use X11 GLFW (XWayland) so GNOME decorates the window; libdecor's GTK
    # plugin crashes on this system.
    os.environ.setdefault('PYGLFW_LIBRARY_VARIANT', 'x11')


if IS_FROZEN:
    # pyglfw loads its native library through ctypes at runtime; its frozen-mode
    # search does not include the bundle root where the lib is packed, so point
    # it at the bundled copy explicitly (checked before any other search path).
    for _glfw_lib in ("libglfw.3.dylib", "glfw3.dll", "libglfw.so.3", "libglfw.so"):
        _glfw_cand = os.path.join(sys._MEIPASS, _glfw_lib)
        if os.path.exists(_glfw_cand):
            os.environ.setdefault("PYGLFW_LIBRARY", _glfw_cand)
            break

    if sys.platform == "darwin":
        # A double-clicked .app starts with cwd="/". Read-only resources are
        # resolved from the bundle via utils.resource_paths, so move the working
        # directory next to the .app; user data (models, presets, recordings,
        # screenshots, training output) is then written there, outside the bundle.
        # Those output folders are created lazily by their features, not here.
        from pathlib import Path
        os.chdir(Path(sys.executable).resolve().parents[3])

import torch

from modules.autolume_live import Autolume


def get_runtime_bin_dir():
    # PyInstaller frozen app: bundled ffmpeg/ffprobe/ninja live in _MEIPASS/bin
    # (a subdir, to avoid colliding with same-named Python packages at the root).
    if IS_FROZEN:
        return os.path.join(sys._MEIPASS, "bin")

    # Development mode
    base = os.path.dirname(os.path.abspath(__file__))
    bin_root = os.path.join(base, "bin")
    for root, dirs, files in os.walk(bin_root):
        if "ffmpeg.exe" in files:
            return root
    return bin_root

BIN_DIR = get_runtime_bin_dir()
os.environ["PATH"] = BIN_DIR + os.pathsep + os.environ.get("PATH", "")


def main():
    app = Autolume()

    while not app.should_close():
        app.draw_frame()
    app.close()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)
    multiprocessing.set_start_method("spawn", force=True)
    main()
