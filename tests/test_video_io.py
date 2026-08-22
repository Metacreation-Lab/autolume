"""Tests for utils.video_io, run against fixtures synthesized with PyAV."""
import av
import numpy as np
import pytest

from utils.video_io import (MediaInfo, VideoIOError, VideoReader, VideoWriter,
                            extract_frames, first_frame, preview_frame, probe)

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


def make_video(path, frames=30, fps=30, width=WIDTH, height=HEIGHT, with_audio=False,
               gop=None):
    options = ({'g': str(gop), 'keyint_min': str(gop), 'sc_threshold': '0'}
               if gop else {})
    with av.open(str(path), mode='w') as container:
        video = container.add_stream('libx264', rate=fps, options=options)
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


def make_box_video(path, frames=30, fps=30):
    """Frames with distinct dHash structure: a white box that moves each frame."""
    with av.open(str(path), mode='w') as container:
        video = container.add_stream('libx264', rate=fps)
        video.width, video.height, video.pix_fmt = WIDTH, HEIGHT, 'yuv420p'
        for i in range(frames):
            image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            x = (i * 11) % (WIDTH - 16)
            image[8:40, x:x + 16] = 255
            frame = av.VideoFrame.from_ndarray(image, format='rgb24')
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode():
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


def test_extract_frames_samples_by_interval(tmp_path):
    path = make_box_video(tmp_path / 'clip.mp4', frames=30, fps=30)  # 1 s long

    written = extract_frames(path, 0.25, str(tmp_path / 'out'), 'clip')

    # candidates at t = 0, 0.25, 0.5, 0.75
    assert written == 4
    names = sorted(p.name for p in (tmp_path / 'out').iterdir())
    assert names == [f'clip_frame_{n:05d}.jpg' for n in range(1, 5)]


def test_extract_frames_interval_zero_takes_every_frame(tmp_path):
    path = make_video(tmp_path / 'clip.mp4', frames=10, fps=30)

    written = extract_frames(path, 0, str(tmp_path / 'out'), 'clip')

    assert written == 10


def test_extract_frames_rejects_negative_interval(tmp_path):
    path = make_video(tmp_path / 'clip.mp4')
    with pytest.raises(VideoIOError):
        extract_frames(path, -1, str(tmp_path / 'out'), 'clip')


def test_extract_frames_reports_written_and_timestamp(tmp_path):
    path = make_box_video(tmp_path / 'clip.mp4', frames=30, fps=30)
    seen = []

    written = extract_frames(path, 0.25, str(tmp_path / 'out'), 'clip',
                             on_progress=lambda n, t: seen.append((n, t)))

    assert written == 4
    assert [n for n, _ in seen] == [1, 2, 3, 4]
    timestamps = [t for _, t in seen]
    assert timestamps == sorted(timestamps)
    assert timestamps[0] == pytest.approx(0.0, abs=0.05)


def test_extract_frames_cancels_early(tmp_path):
    path = make_box_video(tmp_path / 'clip.mp4', frames=30, fps=30)
    seen = []

    written = extract_frames(path, 0, str(tmp_path / 'out'), 'clip',
                             on_progress=lambda n, t: seen.append(n),
                             should_cancel=lambda: len(seen) >= 5)

    assert 5 <= written < 30
    assert len(list((tmp_path / 'out').iterdir())) == written


def test_video_reader_yields_bgr_frames(video):
    with VideoReader(video) as reader:
        assert reader.info.width == WIDTH
        frames = list(reader.frames())

    assert len(frames) == 30
    assert all(f.shape == (HEIGHT, WIDTH, 3) and f.dtype == np.uint8 for f in frames)
    # gradient() ramps red along x and holds blue constant; in bgr24 the ramp
    # must land on channel 2 — a silent RGB/BGR swap would corrupt every
    # consumer's colors while passing shape checks.
    first = frames[0]
    assert int(first[0, -1, 2]) - int(first[0, 0, 2]) > 200
    assert abs(int(first[0, -1, 0]) - int(first[0, 0, 0])) < 30


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
        frames = list(reader.frames())
    assert len(frames) == 30
    # the red x-ramp written on bgr channel 2 must come back on channel 2
    assert int(frames[0][0, -1, 2]) - int(frames[0][0, 0, 2]) > 200


def test_video_writer_accepts_encoder_options(tmp_path):
    out = tmp_path / 'preset.mp4'
    with VideoWriter(str(out), WIDTH, HEIGHT, 30,
                     options={'preset': 'veryfast'}) as writer:
        for i in range(10):
            writer.write(gradient(i)[:, :, ::-1])

    info = probe(str(out))
    assert (info.width, info.height) == (WIDTH, HEIGHT)
    with VideoReader(str(out)) as reader:
        assert len(list(reader.frames())) == 10


def test_video_writer_copies_audio(tmp_path):
    source = make_video(tmp_path / 'source.mp4', with_audio=True)
    out = tmp_path / 'with_audio.mp4'

    with VideoWriter(str(out), WIDTH, HEIGHT, 30, audio_from=source) as writer:
        for i in range(30):
            writer.write(gradient(i)[:, :, ::-1])

    info = probe(str(out))
    assert info.has_audio is True
    # the source carries 1.0s of audio; a bug dropping most packets would
    # still report has_audio, but not the full duration
    assert info.duration == pytest.approx(1.0, abs=0.2)


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


def test_preview_frame_picks_a_middle_frame(tmp_path):
    # 30 frames at 30 fps with keyframes every 10: seeking to t=0.5 lands on
    # the keyframe at frame 10, whose blue channel is 10 * 7 = 70.
    path = make_video(tmp_path / 'clip.mp4', frames=30, fps=30, gop=10)

    frame = preview_frame(path)

    assert frame is not None and frame.shape == (HEIGHT, WIDTH, 3)
    assert abs(int(frame[0, 0, 2]) - 70) <= 20   # not the first frame (blue 0)


def test_preview_frame_returns_none_for_garbage(junk):
    assert preview_frame(junk) is None
