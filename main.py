import multiprocessing
import os
import sys

IS_FROZEN = getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")

if sys.platform == 'darwin':
    # Must be set before torch is imported.
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

if sys.platform == 'linux' and os.environ.get('WAYLAND_DISPLAY'):
    # Use X11 GLFW (XWayland) so GNOME decorates the window; native Wayland
    # requires libdecor-gtk which is unavailable or crashes on some systems.
    os.environ.setdefault('PYGLFW_LIBRARY_VARIANT', 'x11')


from utils.user_data import init_data_root


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
    # Materialise the user data root so it is discoverable even before any
    # feature writes to it. Category subfolders are still created lazily.
    init_data_root()

    from modules.autolume_live import Autolume

    app = Autolume()

    while not app.should_close():
        app.draw_frame()
    app.close()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    multiprocessing.set_start_method("spawn", force=True)

    from utils.app_logging import setup_main_logging, shutdown_logging
    setup_main_logging()

    import logging
    logger = logging.getLogger("autolume")

    try:
        # Imported here (not at module level) so a failure in the heavy
        # dependency stack is captured by the log instead of vanishing in
        # windowed frozen builds.
        import torch

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_grad_enabled(False)

        main()
    except SystemExit:
        raise
    except BaseException:
        logger.critical("Fatal error", exc_info=True)
        raise
    finally:
        shutdown_logging()
