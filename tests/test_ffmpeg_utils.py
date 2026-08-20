import json
import subprocess

import ffmpeg
import pytest

from utils import ffmpeg_utils


def test_probe_parses_json_and_hides_window(monkeypatch):
    seen = {}

    def fake_run(args, capture_output, creationflags):
        seen["args"] = args
        seen["creationflags"] = creationflags
        stdout = json.dumps({"streams": [], "format": {"duration": "5.0"}}).encode()
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr=b"")

    monkeypatch.setattr(ffmpeg_utils.subprocess, "run", fake_run)
    result = ffmpeg_utils.probe("video.mp4")

    assert result["format"]["duration"] == "5.0"
    assert seen["args"] == ["ffprobe", "-show_format", "-show_streams", "-of", "json", "video.mp4"]
    assert seen["creationflags"] == ffmpeg_utils.CREATIONFLAGS


def test_probe_raises_ffmpeg_error_on_failure(monkeypatch):
    def fake_run(args, capture_output, creationflags):
        return subprocess.CompletedProcess(args, 1, stdout=b"", stderr=b"boom")

    monkeypatch.setattr(ffmpeg_utils.subprocess, "run", fake_run)
    with pytest.raises(ffmpeg.Error) as excinfo:
        ffmpeg_utils.probe("video.mp4")
    assert excinfo.value.stderr == b"boom"


def test_run_async_compiles_stream_and_hides_window(monkeypatch):
    seen = {}

    def fake_popen(args, stdin, stdout, creationflags):
        seen.update(args=args, stdin=stdin, stdout=stdout, creationflags=creationflags)
        return "process"

    monkeypatch.setattr(ffmpeg_utils.subprocess, "Popen", fake_popen)
    stream = ffmpeg.input("in.mp4").output("out.mp4", vf="fps=10")
    process = ffmpeg_utils.run_async(stream, pipe_stdout=True)

    assert process == "process"
    assert seen["args"][0] == "ffmpeg"
    assert seen["args"][-1] == "out.mp4"
    assert "fps=10" in seen["args"]
    assert seen["stdin"] is None
    assert seen["stdout"] == subprocess.PIPE
    assert seen["creationflags"] == ffmpeg_utils.CREATIONFLAGS


def test_run_async_pipe_stdin(monkeypatch):
    seen = {}

    def fake_popen(args, stdin, stdout, creationflags):
        seen.update(stdin=stdin, stdout=stdout)
        return "process"

    monkeypatch.setattr(ffmpeg_utils.subprocess, "Popen", fake_popen)
    ffmpeg_utils.run_async(ffmpeg.input("pipe:").output("out.mp4"), pipe_stdin=True)

    assert seen["stdin"] == subprocess.PIPE
    assert seen["stdout"] is None
