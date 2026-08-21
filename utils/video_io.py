"""Video I/O built on PyAV: probing, frame extraction, reading and writing.

This is the only module allowed to import ``av``; everything else in the
codebase goes through the helpers here.
"""
import logging
import os
from dataclasses import dataclass
from fractions import Fraction

import av
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Tolerance when comparing decoded timestamps against emission boundaries.
_EPS = 1e-6


class VideoIOError(Exception):
    """A media file could not be opened, decoded or written."""


@dataclass
class MediaInfo:
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool


def _open(path, mode='r'):
    try:
        return av.open(str(path), mode=mode)
    except (av.FFmpegError, OSError) as e:
        raise VideoIOError(f'Cannot open "{path}": {e}') from e


def _video_stream(container, path):
    stream = next(iter(container.streams.video), None)
    if stream is None:
        raise VideoIOError(f'No video stream in "{path}"')
    return stream


def _media_info(container, stream, path):
    if container.duration is not None:
        duration = container.duration / av.time_base
    elif stream.duration is not None and stream.time_base is not None:
        duration = float(stream.duration * stream.time_base)
    else:
        logger.warning('Unknown duration for "%s"', path)
        duration = 0.0
    return MediaInfo(
        duration=duration,
        width=stream.width or 0,
        height=stream.height or 0,
        fps=float(stream.average_rate) if stream.average_rate else 0.0,
        has_audio=bool(container.streams.audio),
    )


def probe(path):
    """Container/stream metadata of a media file."""
    with _open(path) as container:
        return _media_info(container, _video_stream(container, path), path)


def first_frame(path):
    """First decodable frame as an rgb24 HxWx3 array, or None if unreadable."""
    try:
        with _open(path) as container:
            for frame in container.decode(video=0):
                return frame.to_ndarray(format='rgb24')
        logger.warning('No decodable frame in "%s"', path)
    except (VideoIOError, av.FFmpegError, OSError, ValueError, IndexError) as e:
        logger.warning('Cannot read first frame of "%s": %s', path, e)
    return None


def extract_frames(path, fps, out_dir, name_prefix, on_progress=None, should_cancel=None):
    """Write JPEG frames sampled at ``fps`` into ``out_dir``; return how many.

    Files are named ``{name_prefix}_frame_{n:05d}.jpg`` with ``n`` starting at 1.
    Emission is driven by decoded timestamps, so variable frame rate sources and
    sources slower than the target rate still yield the expected frame count.
    """
    if fps <= 0:
        raise VideoIOError(f'Invalid target fps {fps} for "{path}"')
    os.makedirs(out_dir, exist_ok=True)

    written = 0
    with _open(path) as container:
        stream = _video_stream(container, path)
        stream.thread_type = 'AUTO'
        source_rate = float(stream.average_rate) if stream.average_rate else 0.0
        step = 1.0 / source_rate if source_rate else 1.0 / fps

        def emit(image):
            nonlocal written
            written += 1
            cv2.imwrite(os.path.join(out_dir, f'{name_prefix}_frame_{written:05d}.jpg'),
                        image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if on_progress is not None:
                on_progress(written)

        cancelled = False
        last_image = None
        end_time = 0.0
        try:
            for index, frame in enumerate(container.decode(stream)):
                if should_cancel is not None and should_cancel():
                    cancelled = True
                    break
                if frame.pts is not None and stream.time_base is not None:
                    timestamp = float(frame.pts * stream.time_base)
                elif source_rate:
                    timestamp = index / source_rate
                else:
                    timestamp = written / fps
                duration = (float(frame.duration * stream.time_base)
                            if frame.duration and stream.time_base is not None else step)
                end_time = timestamp + duration
                if timestamp + _EPS < written / fps:
                    continue
                last_image = frame.to_ndarray(format='bgr24')
                while timestamp + _EPS >= written / fps:
                    emit(last_image)
        except av.FFmpegError as e:
            raise VideoIOError(f'Failed to decode "{path}": {e}') from e

        # The last frame stays on screen for its own duration; holding it there
        # keeps the count matching the ffmpeg fps filter on slow sources.
        if not cancelled and last_image is not None:
            while written / fps < end_time - _EPS:
                emit(last_image)
    return written


class VideoReader:
    """Sequential decoder yielding bgr24 frames; usable as a context manager."""

    def __init__(self, path):
        self.path = str(path)
        self._container = _open(self.path)
        try:
            self._stream = _video_stream(self._container, self.path)
            self._stream.thread_type = 'AUTO'
            self.info = _media_info(self._container, self._stream, self.path)
        except VideoIOError:
            self._container.close()
            raise

    def frames(self):
        try:
            for frame in self._container.decode(self._stream):
                yield frame.to_ndarray(format='bgr24')
        except av.FFmpegError as e:
            raise VideoIOError(f'Failed to decode "{self.path}": {e}') from e

    def close(self):
        if self._container is not None:
            self._container.close()
            self._container = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class VideoWriter:
    """H.264/yuv420p mp4 encoder; usable as a context manager.

    ``audio_from`` names a source file whose audio stream, if any, is remuxed
    by packet copy. ``bit_rate`` is in bits per second. ``options`` are passed
    to the libx264 encoder (e.g. {'preset': 'veryfast'}).
    """

    def __init__(self, path, width, height, fps, audio_from=None, bit_rate=None,
                 options=None):
        self.path = str(path)
        if width % 2 or height % 2:
            raise VideoIOError(
                f'Odd frame size {width}x{height} for "{self.path}"; '
                'yuv420p requires even width and height')
        self._container = _open(self.path, mode='w')
        try:
            self._stream = self._container.add_stream(
                'libx264', rate=Fraction(fps).limit_denominator(65535),
                options=options or {})
            self._stream.width = width
            self._stream.height = height
            self._stream.pix_fmt = 'yuv420p'
            if bit_rate:
                self._stream.bit_rate = bit_rate
            if audio_from is not None:
                self._copy_audio(audio_from)
        except av.FFmpegError as e:
            self._container.close()
            self._container = None
            raise VideoIOError(f'Cannot set up "{self.path}": {e}') from e
        except Exception:
            self._container.close()
            self._container = None
            raise

    def _copy_audio(self, source):
        # The audio stream must exist before the first video packet is muxed.
        # Muxing all audio up front keeps the writer simple; the muxer holds the
        # compressed audio (small) until video timestamps catch up.
        with _open(source) as container:
            stream = next(iter(container.streams.audio), None)
            if stream is None:
                return
            out_stream = self._container.add_stream_from_template(stream)
            for packet in container.demux(stream):
                if packet.dts is None:
                    continue
                packet.stream = out_stream
                self._container.mux(packet)

    def write(self, frame_bgr):
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame_bgr), format='bgr24')
        try:
            for packet in self._stream.encode(frame):
                self._container.mux(packet)
        except av.FFmpegError as e:
            raise VideoIOError(f'Failed to encode "{self.path}": {e}') from e

    def close(self):
        if self._container is None:
            return
        try:
            for packet in self._stream.encode():
                self._container.mux(packet)
        except av.FFmpegError as e:
            raise VideoIOError(f'Failed to finalize "{self.path}": {e}') from e
        finally:
            self._container.close()
            self._container = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
