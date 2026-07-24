import logging

from autolume.live.runtime import build_runtime
from autolume.live.ui.window import run_ui


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    runtime = build_runtime()
    runtime.start()
    try:
        run_ui(runtime)
    finally:
        runtime.stop()


if __name__ == "__main__":
    main()
