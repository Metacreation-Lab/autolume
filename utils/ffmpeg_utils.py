"""Windowless launchers for ffmpeg/ffprobe.

ffmpeg-python starts its subprocesses with a bare Popen; from the windowed
release build every such spawn allocates a new console, which on Windows 11
opens a Windows Terminal window per probe/encode (and those can outlive the
process). All runtime ffmpeg/ffprobe launches must go through these helpers.
"""

import json
import os
import subprocess

import ffmpeg

CREATIONFLAGS = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def probe(filename, cmd="ffprobe"):
    """Drop-in for ffmpeg.probe() minus the console window on Windows."""
    args = [cmd, "-show_format", "-show_streams", "-of", "json", filename]
    p = subprocess.run(args, capture_output=True, creationflags=CREATIONFLAGS)
    if p.returncode != 0:
        raise ffmpeg.Error("ffprobe", p.stdout, p.stderr)
    return json.loads(p.stdout.decode("utf-8"))


def run_async(stream, pipe_stdin=False, pipe_stdout=False):
    """Drop-in for stream.run_async() minus the console window on Windows."""
    return subprocess.Popen(
        ffmpeg.compile(stream),
        stdin=subprocess.PIPE if pipe_stdin else None,
        stdout=subprocess.PIPE if pipe_stdout else None,
        creationflags=CREATIONFLAGS,
    )
