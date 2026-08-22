"""Tests for the dataset preprocessing video path: media probing and extraction."""
import queue
import threading
import time

import numpy as np
import pytest

from modules.preprocessing_module import _VideoInfoProber
from utils import video_io
from utils.dataset_preprocessing_utils import DatasetPreprocessingUtils
from utils.video_io import MediaInfo, VideoIOError, VideoWriter

WIDTH, HEIGHT = 64, 48
SOURCE_FPS = 20


def make_video(path, seconds=2):
    """Synthesize a small mp4 of distinct textured frames; returns str path."""
    with VideoWriter(str(path), WIDTH, HEIGHT, SOURCE_FPS) as writer:
        for i in range(seconds * SOURCE_FPS):
            frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            x = (i * 7) % (WIDTH - 12)
            frame[8:40, x:x + 12] = 255
            writer.write(frame)
    return str(path)


def run_extraction(paths, interval, queue_in=None):
    """Run extract_videos synchronously and return the emitted messages."""
    queue_out = queue.Queue()
    DatasetPreprocessingUtils.extract_videos(paths, interval,
                                             queue_in or queue.Queue(),
                                             queue_out)
    messages = []
    while not queue_out.empty():
        messages.append(queue_out.get_nowait())
    return messages


def media_info(duration, fps=20.0):
    return MediaInfo(duration=duration, width=0, height=0, fps=fps, has_audio=False)


@pytest.fixture
def prober():
    instance = _VideoInfoProber()
    yield instance
    instance.shutdown()


def collect(prober, expected, timeout=10.0):
    """Poll until ``expected`` infos arrived or the timeout expires."""
    deadline = time.monotonic() + timeout
    infos = {}
    while len(infos) < expected and time.monotonic() < deadline:
        infos.update(prober.poll())
        time.sleep(0.01)
    return infos


# --- Video info prober -------------------------------------------------------

def test_prober_delivers_info(monkeypatch, prober):
    monkeypatch.setattr(video_io, 'probe', lambda path: media_info(len(path)))

    prober.start(['a.mp4', 'bb.mp4'])

    assert collect(prober, 2) == {'a.mp4': media_info(5.0),
                                  'bb.mp4': media_info(6.0)}


def test_prober_poll_is_empty_without_work(prober):
    assert prober.poll() == {}


def test_prober_failed_probe_yields_empty_info(monkeypatch, prober):
    def failing(path):
        raise VideoIOError(f'Cannot open "{path}"')

    monkeypatch.setattr(video_io, 'probe', failing)
    prober.start(['broken.mp4'])

    assert collect(prober, 1) == {'broken.mp4': media_info(0.0, fps=0.0)}


def test_prober_cancel_discards_stale_results(monkeypatch, prober):
    released = threading.Event()

    def blocking(path):
        released.wait(10)
        return media_info(1.0)

    monkeypatch.setattr(video_io, 'probe', blocking)
    prober.start(['stale.mp4'])
    prober.cancel()
    released.set()

    assert collect(prober, 1, timeout=1.0) == {}


def test_prober_restart_delivers_only_the_new_generation(monkeypatch, prober):
    monkeypatch.setattr(video_io, 'probe', lambda path: media_info(3.0))

    prober.start(['old.mp4'])
    prober.cancel()
    prober.start(['new.mp4'])

    assert collect(prober, 1) == {'new.mp4': media_info(3.0)}


# --- Frame count estimate ----------------------------------------------------

def test_estimate_uses_duration_over_interval():
    info = MediaInfo(duration=60.0, width=0, height=0, fps=20.0, has_audio=False)
    assert DatasetPreprocessingUtils.estimate_extracted_frames(info, 10.0) == 7


def test_estimate_interval_zero_counts_every_frame():
    info = MediaInfo(duration=2.0, width=0, height=0, fps=20.0, has_audio=False)
    assert DatasetPreprocessingUtils.estimate_extracted_frames(info, 0) == 40


def test_estimate_unknown_duration_is_zero():
    info = MediaInfo(duration=0.0, width=0, height=0, fps=20.0, has_audio=False)
    assert DatasetPreprocessingUtils.estimate_extracted_frames(info, 1.0) == 0


# --- Frame extraction --------------------------------------------------------

def test_extract_videos_writes_named_frames(tmp_path):
    source = make_video(tmp_path / 'clip.mp4')  # 2 s at 20 fps

    messages = run_extraction([source], 0.5)

    out_dir = tmp_path / 'clip_frames @ 0.5s'
    names = sorted(p.name for p in out_dir.iterdir())
    assert names[0] == 'clip_frame_00001.jpg'
    assert len(names) == 4      # candidates at t = 0, 0.5, 1.0, 1.5
    assert messages[-1] == {'type': 'completed', 'results': [str(out_dir)]}


def test_extract_videos_reports_monotonic_progress(tmp_path):
    sources = [make_video(tmp_path / 'one.mp4'), make_video(tmp_path / 'two.mp4')]

    messages = run_extraction(sources, 0.1)

    progress = [m for m in messages if m['type'] == 'progress']
    percentages = [m['percentage'] for m in progress]
    assert percentages == sorted(percentages)
    assert percentages[0] == 0.0
    assert percentages[-1] == pytest.approx(100.0, abs=1.0)
    assert all(m['total'] == 2 for m in progress)
    assert {m['current_file'] for m in progress} == {'one.mp4', 'two.mp4'}


def test_extract_videos_skips_undecodable_video(tmp_path):
    junk = tmp_path / 'junk.mp4'
    junk.write_bytes(b'not a video at all' * 64)
    source = make_video(tmp_path / 'good.mp4')

    messages = run_extraction([str(junk), source], 0.5)

    assert messages[-1] == {'type': 'completed',
                            'results': [str(tmp_path / 'good_frames @ 0.5s')]}


def test_extract_videos_cancels_mid_video(tmp_path):
    source = make_video(tmp_path / 'clip.mp4', seconds=3)

    class CancelAfter:
        """queue_in stand-in that answers cancel once probed often enough.

        extract_videos polls once per sampled candidate; if that cadence is
        throttled, adjust the probe budget rather than the design.
        """

        def __init__(self, probes):
            self.probes = probes

        def empty(self):
            self.probes -= 1
            return self.probes > 0

        def get_nowait(self):
            return 'cancel'

    messages = run_extraction([source], 0.1, queue_in=CancelAfter(4))

    written = list((tmp_path / 'clip_frames @ 0.1s').iterdir())
    assert 0 < len(written) < 30
    assert all(m['type'] == 'progress' for m in messages)
