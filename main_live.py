"""Entry point for the live runtime, the ``Autolume Live`` executable.

A file at the repository root because PyInstaller builds from a script path;
everything it needs lives in ``autolume.live.__main__`` so that the frozen app
and ``python -m autolume.live`` start the same way.
"""
import multiprocessing

from autolume.live.__main__ import run

if __name__ == "__main__":
    multiprocessing.freeze_support()
    run()
