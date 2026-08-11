import multiprocessing
import os
import sys

IS_FROZEN = getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")

# Build cache for the custom CUDA ops: precompiled entries ship inside the
# bundle; in development the cache lives at the repository root. Must be set
# before the first op is loaded and is inherited by spawned worker processes.
os.environ.setdefault(
    "TORCH_EXTENSIONS_DIR",
    os.path.join(sys._MEIPASS, "torch_extensions") if IS_FROZEN
    else os.path.join(os.path.dirname(os.path.abspath(__file__)), "torch_extensions"),
)

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

if IS_FROZEN:
    # imageio-ffmpeg's own binary is pruned from the bundle (release.py);
    # route it to the ffmpeg already shipped in bin/.
    os.environ.setdefault(
        "IMAGEIO_FFMPEG_EXE",
        os.path.join(BIN_DIR, "ffmpeg.exe" if os.name == "nt" else "ffmpeg"),
    )


def main():
    # Materialise the user data root so it is discoverable even before any
    # feature writes to it. Category subfolders are still created lazily.
    init_data_root()

    # Loads the heavy dependency stack (torch, rendering) on a
    # background thread while the splash screen shows its progress. Started
    # by the splash once it is on screen: imports contend for the GIL, so
    # starting earlier delays the first frame on cold systems.
    from utils.startup_loader import StartupLoader
    loader = StartupLoader()

    # Non-blocking check for a newer GitHub release; result shows in Settings.
    from utils.update_check import start_update_check
    start_update_check()

    from modules.autolume_live import Autolume

    app = Autolume(loader)

    while not app.should_close():
        app.draw_frame()
    app.close()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    multiprocessing.set_start_method("spawn", force=True)

    from utils.app_logging import setup_main_logging, shutdown_logging
    setup_main_logging()

    from utils import crash_report
    crash_report.process_startup()

    from utils import opengl_support

    import logging
    logger = logging.getLogger("autolume")

    try:
        main()
        crash_report.mark_clean_exit()
    except SystemExit:
        crash_report.mark_clean_exit()
        raise
    except KeyboardInterrupt:
        crash_report.mark_clean_exit()
        raise
    except opengl_support.OpenGLUnsupportedError as err:
        # Not a crash: tell the user what their machine is missing instead of
        # offering a crash report.
        logger.critical("Unsupported graphics environment: %s", err)
        crash_report.mark_clean_exit()
        opengl_support.show_unsupported_dialog(str(err))
        sys.exit(1)
    except BaseException:
        logger.critical("Fatal error", exc_info=True)
        crash_report.handle_fatal_exception()
        raise
    finally:
        shutdown_logging()
