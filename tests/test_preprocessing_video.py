"""Tests for the dataset preprocessing video path: duration probing and extraction."""
import queue
import threading
import time

import numpy as np
import pytest

from modules.preprocessing_module import _DurationProber
from utils import video_io
from utils.dataset_preprocessing_utils import DatasetPreprocessingUtils
from utils.video_io import MediaInfo, VideoIOError, VideoWriter

WIDTH, HEIGHT = 64, 48
SOURCE_FPS = 20


def make_video(path, seconds=2):
    """Synthesize a small mp4 and return its path as a string."""
    with VideoWriter(str(path), WIDTH, HEIGHT, SOURCE_FPS) as writer:
        for i in range(seconds * SOURCE_FPS):
            frame = np.full((HEIGHT, WIDTH, 3), i * 7 % 256, dtype=np.uint8)
            writer.write(frame)
    return str(path)


def run_extraction(paths, fps, queue_in=None):
    """Run extract_videos synchronously and return the emitted messages."""
    queue_out = queue.Queue()
    DatasetPreprocessingUtils.extract_videos(paths, fps, queue_in or queue.Queue(), queue_out)
    messages = []
    while not queue_out.empty():
        messages.append(queue_out.get_nowait())
    return messages


def info(duration):
    return MediaInfo(duration=duration, width=WIDTH, height=HEIGHT, fps=SOURCE_FPS,
                     has_audio=False)


@pytest.fixture
def prober():
    instance = _DurationProber()
    yield instance
    instance.shutdown()


def collect(prober, expected, timeout=10.0):
    """Poll until ``expected`` durations arrived or the timeout expires."""
    deadline = time.monotonic() + timeout
    durations = {}
    while len(durations) < expected and time.monotonic() < deadline:
        durations.update(prober.poll())
        time.sleep(0.01)
    return durations


# --- Duration prober ---------------------------------------------------------

def test_prober_delivers_durations(monkeypatch, prober):
    monkeypatch.setattr(video_io, 'probe', lambda path: info(len(path)))

    prober.start(['a.mp4', 'bb.mp4'])

    assert collect(prober, 2) == {'a.mp4': 5.0, 'bb.mp4': 6.0}


def test_prober_poll_is_empty_without_work(prober):
    assert prober.poll() == {}


def test_prober_failed_probe_yields_zero(monkeypatch, prober):
    def failing(path):
        raise VideoIOError(f'Cannot open "{path}"')

    monkeypatch.setattr(video_io, 'probe', failing)
    prober.start(['broken.mp4'])

    assert collect(prober, 1) == {'broken.mp4': 0.0}


def test_prober_cancel_discards_stale_results(monkeypatch, prober):
    released = threading.Event()

    def blocking(path):
        released.wait(10)
        return info(1.0)

    monkeypatch.setattr(video_io, 'probe', blocking)
    prober.start(['stale.mp4'])
    prober.cancel()
    released.set()

    assert collect(prober, 1, timeout=1.0) == {}


def test_prober_restart_delivers_only_the_new_generation(monkeypatch, prober):
    monkeypatch.setattr(video_io, 'probe', lambda path: info(3.0))

    prober.start(['old.mp4'])
    prober.cancel()
    prober.start(['new.mp4'])

    assert collect(prober, 1) == {'new.mp4': 3.0}


# --- Frame extraction --------------------------------------------------------

def test_extract_videos_writes_named_frames(tmp_path):
    source = make_video(tmp_path / 'clip.mp4')

    messages = run_extraction([source], 5)

    out_dir = tmp_path / 'clip_frames @ 5 fps'
    names = sorted(p.name for p in out_dir.iterdir())
    assert names[0] == 'clip_frame_00001.jpg'
    assert len(names) == 10
    assert messages[-1] == {'type': 'completed', 'results': [str(out_dir)]}


def test_extract_videos_reports_monotonic_progress(tmp_path):
    sources = [make_video(tmp_path / 'one.mp4'), make_video(tmp_path / 'two.mp4')]

    messages = run_extraction(sources, 10)

    progress = [m for m in messages if m['type'] == 'progress']
    percentages = [m['percentage'] for m in progress]
    assert percentages == sorted(percentages)
    assert percentages[0] == 0.0
    assert percentages[-1] == pytest.approx(100.0, abs=1.0)
    assert all(m['total'] == 2 for m in progress)
    assert {m['current_file'] for m in progress} == {'one.mp4', 'two.mp4'}
    assert progress[-1]['current'] == 1


def test_extract_videos_skips_undecodable_video(tmp_path):
    junk = tmp_path / 'junk.mp4'
    junk.write_bytes(b'not a video at all' * 64)
    source = make_video(tmp_path / 'good.mp4')

    messages = run_extraction([str(junk), source], 5)

    assert messages[-1] == {'type': 'completed',
                            'results': [str(tmp_path / 'good_frames @ 5 fps')]}


def test_extract_videos_cancels_mid_video(tmp_path):
    source = make_video(tmp_path / 'clip.mp4', seconds=3)

    class CancelAfter:
        """queue_in stand-in that answers "cancel" once probed often enough.

        Counting empty() calls couples this to how often extract_videos polls
        for cancellation (currently once per decoded frame); if that cadence
        is ever throttled, adjust the probe budget rather than the design.
        """

        def __init__(self, probes):
            self.probes = probes

        def empty(self):
            self.probes -= 1
            return self.probes > 0

        def get_nowait(self):
            return 'cancel'

    messages = run_extraction([source], 20, queue_in=CancelAfter(10))

    written = list((tmp_path / 'clip_frames @ 20 fps').iterdir())
    assert 0 < len(written) < 60
    assert all(m['type'] == 'progress' for m in messages)
