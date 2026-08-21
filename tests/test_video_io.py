"""Tests for utils.video_io, run against fixtures synthesized with PyAV."""
import av
import numpy as np
import pytest

from utils.video_io import (MediaInfo, VideoIOError, VideoReader, VideoWriter,
                            extract_frames, first_frame, probe)

WIDTH, HEIGHT = 64, 48
SAMPLE_RATE = 48000


def gradient(index, width=WIDTH, height=HEIGHT):
    """Deterministic rgb24 test pattern that changes with the frame index."""
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = x[None, :]
    image[:, :, 1] = y[:, None]
    image[:, :, 2] = index * 7 % 256
    return image


def make_video(path, frames=30, fps=30, width=WIDTH, height=HEIGHT, with_audio=False):
    with av.open(str(path), mode='w') as container:
        video = container.add_stream('libx264', rate=fps)
        video.width = width
        video.height = height
        video.pix_fmt = 'yuv420p'
        audio = container.add_stream('aac', rate=SAMPLE_RATE) if with_audio else None

        for i in range(frames):
            frame = av.VideoFrame.from_ndarray(gradient(i, width, height), format='rgb24')
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)

        if audio is not None:
            count = int(SAMPLE_RATE * frames / fps)
            t = np.arange(count) / SAMPLE_RATE
            samples = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16).reshape(1, -1)
            frame = av.AudioFrame.from_ndarray(samples, format='s16', layout='mono')
            frame.sample_rate = SAMPLE_RATE
            for packet in audio.encode(frame):
                container.mux(packet)
            for packet in audio.encode():
                container.mux(packet)
    return str(path)


@pytest.fixture
def video(tmp_path):
    return make_video(tmp_path / 'clip.mp4')


@pytest.fixture
def junk(tmp_path):
    path = tmp_path / 'junk.mp4'
    path.write_bytes(b'not a video, just some bytes' * 64)
    return str(path)


def test_probe_reports_stream_properties(video):
    info = probe(video)
    assert isinstance(info, MediaInfo)
    assert (info.width, info.height) == (WIDTH, HEIGHT)
    assert info.fps == pytest.approx(30.0)
    assert info.duration == pytest.approx(1.0, abs=0.1)
    assert info.has_audio is False


def test_probe_detects_audio(tmp_path):
    path = make_video(tmp_path / 'with_audio.mp4', with_audio=True)
    assert probe(path).has_audio is True


def test_probe_missing_file_raises_with_path(tmp_path):
    missing = str(tmp_path / 'nope.mp4')
    with pytest.raises(VideoIOError) as excinfo:
        probe(missing)
    assert missing in str(excinfo.value)


def test_probe_non_video_raises(junk):
    with pytest.raises(VideoIOError):
        probe(junk)


def test_extract_frames_count_and_naming(tmp_path):
    path = make_video(tmp_path / 'two_seconds.mp4', frames=60, fps=30)
    out_dir = tmp_path / 'frames'

    written = extract_frames(path, 10, str(out_dir), 'clip')

    assert written == 20
    names = sorted(p.name for p in out_dir.iterdir())
    assert names[0] == 'clip_frame_00001.jpg'
    assert names[-1] == 'clip_frame_00020.jpg'
    assert len(names) == 20


def test_extract_frames_upsamples_slow_source(tmp_path):
    path = make_video(tmp_path / 'slow.mp4', frames=10, fps=5)
    out_dir = tmp_path / 'frames'

    written = extract_frames(path, 10, str(out_dir), 'slow')

    assert written == 20
    assert len(list(out_dir.iterdir())) == 20


def test_extract_frames_reports_progress(tmp_path, video):
    seen = []
    written = extract_frames(video, 10, str(tmp_path / 'out'), 'clip',
                             on_progress=seen.append)

    assert seen == list(range(1, written + 1))


def test_extract_frames_cancels_early(tmp_path):
    path = make_video(tmp_path / 'long.mp4', frames=90, fps=30)
    out_dir = tmp_path / 'out'
    seen = []

    written = extract_frames(path, 30, str(out_dir), 'long',
                             on_progress=seen.append,
                             should_cancel=lambda: len(seen) >= 5)

    assert 5 <= written < 90
    assert len(list(out_dir.iterdir())) == written


def test_video_reader_yields_bgr_frames(video):
    with VideoReader(video) as reader:
        assert reader.info.width == WIDTH
        frames = list(reader.frames())

    assert len(frames) == 30
    assert all(f.shape == (HEIGHT, WIDTH, 3) and f.dtype == np.uint8 for f in frames)


def test_video_reader_rejects_non_video(junk):
    with pytest.raises(VideoIOError):
        VideoReader(junk)


def test_video_writer_roundtrip(tmp_path):
    out = tmp_path / 'written.mp4'
    with VideoWriter(str(out), WIDTH, HEIGHT, 15) as writer:
        for i in range(30):
            writer.write(gradient(i)[:, :, ::-1])

    info = probe(str(out))
    assert (info.width, info.height) == (WIDTH, HEIGHT)
    assert info.fps == pytest.approx(15.0)
    assert info.duration == pytest.approx(2.0, abs=0.2)
    assert info.has_audio is False

    with VideoReader(str(out)) as reader:
        assert len(list(reader.frames())) == 30


def test_video_writer_copies_audio(tmp_path):
    source = make_video(tmp_path / 'source.mp4', with_audio=True)
    out = tmp_path / 'with_audio.mp4'

    with VideoWriter(str(out), WIDTH, HEIGHT, 30, audio_from=source) as writer:
        for i in range(30):
            writer.write(gradient(i)[:, :, ::-1])

    assert probe(str(out)).has_audio is True


def test_video_writer_ignores_silent_audio_source(tmp_path, video):
    out = tmp_path / 'silent.mp4'
    with VideoWriter(str(out), WIDTH, HEIGHT, 30, audio_from=video) as writer:
        writer.write(gradient(0)[:, :, ::-1])

    assert probe(str(out)).has_audio is False


def test_video_writer_rejects_odd_dimensions(tmp_path):
    out = tmp_path / 'odd.mp4'
    with pytest.raises(VideoIOError) as excinfo:
        VideoWriter(str(out), 65, HEIGHT, 30)
    assert str(out) in str(excinfo.value)
    assert not out.exists()


def test_first_frame_returns_rgb_array(video):
    frame = first_frame(video)
    assert frame is not None
    assert frame.shape == (HEIGHT, WIDTH, 3)
    assert frame.dtype == np.uint8
    # Red channel ramps left to right in the synthesized gradient.
    assert int(frame[0, -1, 0]) > int(frame[0, 0, 0])


def test_first_frame_returns_none_for_garbage(junk):
    assert first_frame(junk) is None


def test_first_frame_returns_none_for_missing_file(tmp_path):
    assert first_frame(str(tmp_path / 'nope.mp4')) is None
