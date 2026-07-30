import multiprocessing

from utils import startup_env

# Everything below imports torch, glfw or imgui, all of which read the
# environment as they load, so this call comes first. The GLFW argument is the
# legacy app's half of the two-app policy recorded in release.py.
startup_env.configure(startup_env.PYGLFW)

from utils.user_data import init_data_root


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
    except BaseException:
        logger.critical("Fatal error", exc_info=True)
        crash_report.handle_fatal_exception()
        raise
    finally:
        shutdown_logging()
