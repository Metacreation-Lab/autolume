"""Tests for the headless thumbnail decode path of widgets.thumbnail_widget."""
import numpy as np
import pytest

from utils.video_io import VideoWriter
from widgets.thumbnail_widget import _render_thumbnail

WIDTH, HEIGHT = 64, 48
SIZE = 32


@pytest.fixture
def video(tmp_path):
    path = tmp_path / 'clip.mp4'
    with VideoWriter(path, WIDTH, HEIGHT, fps=10) as writer:
        for i in range(5):
            frame = np.full((HEIGHT, WIDTH, 3), i * 20, dtype=np.uint8)
            writer.write(frame)
    return str(path)


def test_render_thumbnail_video(video):
    thumbnail = _render_thumbnail(video, SIZE, padding=0)
    assert thumbnail.shape == (SIZE, SIZE, 3)
    assert thumbnail.dtype == np.uint8


def test_render_thumbnail_unreadable_video(tmp_path):
    path = tmp_path / 'junk.mp4'
    path.write_bytes(b'not a video, just some bytes' * 64)
    assert _render_thumbnail(str(path), SIZE, padding=0) is None


def test_set_badges_stores_and_clears():
    from widgets.thumbnail_widget import ThumbnailWidget
    w = ThumbnailWidget()
    assert w.badges == {}
    w.set_badges({"a.png": "tip"})
    assert w.badges == {"a.png": "tip"}
    w.set_badges(None)
    assert w.badges == {}


def test_cleanup_clears_badges():
    from widgets.thumbnail_widget import ThumbnailWidget
    w = ThumbnailWidget()
    w.set_badges({"a.png": "tip"})
    w.cleanup()
    assert w.badges == {}
