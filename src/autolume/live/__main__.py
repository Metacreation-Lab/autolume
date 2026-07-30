import logging

from utils import startup_env

# Before the imports below, which pull in torch and imgui_bundle: both read
# the environment as they load. The GLFW argument is the live app's half of
# the two-app policy recorded in release.py.
startup_env.configure(startup_env.IMGUI_BUNDLE)

from utils.user_data import init_data_root

from autolume.live.runtime import build_runtime
from autolume.live.ui.window import run_ui


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    # Materialise the user data root so it is discoverable even before any
    # feature writes to it. Category subfolders are still created lazily.
    init_data_root()
    runtime = build_runtime()
    runtime.start()
    try:
        run_ui(runtime)
    finally:
        runtime.stop()


def run() -> None:
    """Entry point for both the ``-m`` invocation and the frozen executable.

    Mirrors ``main.py``: the file log is the only place a windowed build can
    report anything, and the crash paths are the shared ones in
    ``utils.crash_report``.
    """
    from utils.app_logging import setup_main_logging, shutdown_logging
    setup_main_logging()

    from utils import crash_report
    crash_report.process_startup()

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


if __name__ == "__main__":
    run()
