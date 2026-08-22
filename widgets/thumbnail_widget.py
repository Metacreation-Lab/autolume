import logging
import os
import sys
import math
import collections
import queue

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import imgui
import PIL.Image
import PIL.ImageOps

from utils import video_io
from utils.gui_utils import gl_utils

logger = logging.getLogger(__name__)
from utils.dataset_preprocessing_utils import DatasetPreprocessingUtils

# Threading and caching parameters
WORKERS = min(4, max(1, os.cpu_count() or 1))
MAX_IN_FLIGHT = WORKERS * 2
CACHE_LIMIT = 5000
BUFFER_ROWS = 50
UPLOADS_PER_FRAME = 16

# Grid style
THUMB_MIN = 120
THUMB_MAX = 220
GAP_X = 32
GAP_Y = 16

cv2.setNumThreads(1)  # we parallelize across threads ourselves


def _render_thumbnail(file_path, size, padding):
    """Decode an image/video file into a square ``size`` x ``size`` RGB array.

    Non-square images are centered on a square canvas filled with ``padding``.
    Runs on a worker thread. Returns None on failure.
    """
    video_ext = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm')
    try:
        if file_path.lower().endswith(video_ext):
            img = video_io.first_frame(file_path)
        else:
            # Decode JPEGs at reduced resolution: draft() lets libjpeg downsample
            # while decoding (e.g. a 4K source straight to ~1/8 size), a big win
            # since we only need a small thumbnail. No-op for non-JPEG formats.
            # Done here, not in load_images, so that shared utility stays generic.
            pil = PIL.Image.open(file_path)
            pil.draft('RGB', (size, size))
            pil = PIL.ImageOps.exif_transpose(pil)
            img = DatasetPreprocessingUtils().load_images(pil)

        if img is None:
            return None

        h, w = img.shape[:2]
        side = max(h, w)
        channels = img.shape[2] if img.ndim > 2 else 1
        canvas = np.full((side, side, channels), padding, dtype=img.dtype)
        y, x = (side - h) // 2, (side - w) // 2
        canvas[y:y + h, x:x + w] = img
        return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    except Exception as e:
        logger.warning("Error rendering thumbnail for %s: %s", file_path, e)
        return None


class ThumbnailWidget:
    """Lazy, virtualized grid of image thumbnails.

    Each frame we draw the visible cells, request decodes for any we are missing
    (middle-out), and upload finished decodes as textures. Decoded textures are
    held in a fixed-size LRU; cells not yet decoded show a flat loading square.
    """

    def __init__(self, padding_value=0):
        self.thumbnail_size = 140
        self.padding_value = padding_value
        self.selected_files = []

        self.selected_indices = []
        self.last_selected_idx = None
        self.delete_pressed = False

        self._textures = collections.OrderedDict()  # path -> Texture, recent at end (LRU)
        self._futures = {}                           # path -> Future for decodes in flight
        self._done = queue.Queue()                   # (generation, path, ndarray|None)
        self._generation = 0                         # bumped when the file list changes
        self._trash = []                             # textures to delete next frame (GL-safe)
        self._executor = None

    # --- Public API ---------------------------------------------------------

    def update_thumbnails(self, file_paths):
        """Replace the file list. Textures for files that are gone are freed."""
        self.selected_files = file_paths
        self._generation += 1          # in-flight results for the old list become stale
        self._cancel_all_futures()
        keep = set(file_paths)
        for path in [p for p in self._textures if p not in keep]:
            self._trash.append(self._textures.pop(path))

    def render_thumbnails(self, available_width, available_height):
        self._flush_trash()

        files = self.selected_files
        if not files:
            self._draw_empty_message(available_width, available_height)
            return

        per_row, thumb, row_h, left = self._grid_metrics(available_width)
        rows = math.ceil(len(files) / per_row)

        ox, oy = imgui.get_cursor_pos()
        # Reserve the full grid height once so the scroll area never changes
        # size between frames -- this is what stops the drag-scroll jitter.
        imgui.dummy(available_width, rows * row_h)

        self._handle_shortcuts()

        scroll_y = imgui.get_scroll_y()
        first_row = max(0, int(scroll_y / row_h) - BUFFER_ROWS)
        last_row = min(rows, int((scroll_y + available_height) / row_h) + 1 + BUFFER_ROWS)
        first, last = first_row * per_row, min(len(files), last_row * per_row)

        for idx in range(first, last):
            row, col = divmod(idx, per_row)
            x = ox + left + col * (thumb + GAP_X)
            y = oy + row * row_h
            self._draw_thumbnail(idx, files[idx], x, y, thumb)

        self._request_decodes(files, first, last)
        self._evict()

    def poll(self):
        """Upload finished decodes as textures (bounded per frame)."""
        for _ in range(UPLOADS_PER_FRAME):
            try:
                generation, path, data = self._done.get_nowait()
            except queue.Empty:
                return
            self._futures.pop(path, None)
            if generation != self._generation or data is None:
                continue
            texture = self._make_texture(data)
            if texture is not None:
                self._textures[path] = texture
                self._textures.move_to_end(path)

    def cleanup(self):
        """Stop workers and free every texture."""
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        self._drain_done()
        self._cancel_all_futures()
        for texture in list(self._textures.values()) + self._trash:
            self._delete(texture)
        self._textures.clear()
        self._trash.clear()
        self.selected_files = []
        self.selected_indices = []
        self.last_selected_idx = None

    # --- Selection ----------------------------------------------------------

    def select_all(self):
        self.selected_indices = list(range(len(self.selected_files)))
        self.last_selected_idx = None

    def clear_selected(self):
        self.selected_indices = []
        self.last_selected_idx = None

    def get_selected_indices(self):
        if self.selected_indices:
            return sorted(set(self.selected_indices))
        if self.last_selected_idx is not None:
            return [self.last_selected_idx]
        return []

    def is_delete_pressed(self):
        """Return True once if delete was pressed since the last call."""
        pressed, self.delete_pressed = self.delete_pressed, False
        return pressed

    # --- Drawing ------------------------------------------------------------

    def _grid_metrics(self, available_width):
        """Return (per_row, thumb_size, row_height, left_margin) for the grid."""
        inner = available_width - 12
        per_row = max(1, int((inner + GAP_X) // (THUMB_MIN + GAP_X)))
        thumb = min(THUMB_MAX, max(THUMB_MIN, int((inner - (per_row - 1) * GAP_X) // per_row)))
        row_width = per_row * thumb + (per_row - 1) * GAP_X
        left = 6 + max(0, (inner - row_width) // 2)
        row_height = thumb + GAP_Y + imgui.get_text_line_height_with_spacing()
        return per_row, thumb, row_height, left

    def _draw_thumbnail(self, idx, path, x, y, thumb):
        imgui.set_cursor_pos((x, y))
        imgui.push_id(str(idx))

        sx, sy = imgui.get_cursor_screen_pos()
        if imgui.invisible_button("t", thumb, thumb):
            self._select(idx)
        hovered = imgui.is_item_hovered()
        selected = (idx in self.selected_indices
                    or (not self.selected_indices and idx == self.last_selected_idx))

        draw = imgui.get_window_draw_list()
        texture = self._textures.get(path)
        if texture is not None and texture.gl_id is not None:
            self._textures.move_to_end(path)       # mark recently seen for the LRU
            imgui.set_cursor_pos((x, y))
            imgui.image(int(texture.gl_id), thumb, thumb)
        else:
            draw.add_rect_filled(sx, sy, sx + thumb, sy + thumb,
                                 imgui.get_color_u32_rgba(0.18, 0.18, 0.18, 1.0), rounding=6)

        if hovered or selected:
            color = (0.8, 0.4, 0.1, 1.0) if selected else (1.0, 0.8, 0.2, 0.7)
            draw.add_rect(sx, sy, sx + thumb, sy + thumb,
                          imgui.get_color_u32_rgba(*color), rounding=6,
                          thickness=3 if selected else 2)

        name = os.path.basename(path)
        if len(name) > 15:
            name = name[:12] + "..."
        text_x = x + (thumb - imgui.calc_text_size(name)[0]) / 2
        imgui.set_cursor_pos((text_x, y + thumb))
        imgui.text(name)

        imgui.pop_id()

    def _draw_empty_message(self, width, height):
        message = "No images imported"
        text_width = imgui.calc_text_size(message)[0]
        imgui.set_cursor_pos(((width - text_width) / 2,
                              (height - imgui.get_text_line_height()) / 2))
        imgui.text_colored(message, 0.5, 0.5, 0.5, 1.0)

    def _shortcut_mod_down(self):
        # Cmd on macOS, Ctrl elsewhere.
        io = imgui.get_io()
        return io.key_super if sys.platform == 'darwin' else io.key_ctrl

    def _select(self, idx):
        mod = self._shortcut_mod_down()
        shift = imgui.is_key_down(340) or imgui.is_key_down(344)
        if shift and self.last_selected_idx is not None:
            lo, hi = sorted((self.last_selected_idx, idx))
            self.selected_indices = list(range(lo, hi + 1))
        elif mod:
            if idx in self.selected_indices:
                self.selected_indices.remove(idx)
            else:
                self.selected_indices.append(idx)
            self.last_selected_idx = idx
        else:
            self.selected_indices = [idx]
            self.last_selected_idx = idx

    def _handle_shortcuts(self):
        if self._shortcut_mod_down() and imgui.is_key_pressed(65):  # Cmd/Ctrl+A
            self.select_all()
        if imgui.is_key_pressed(261) or imgui.is_key_pressed(259):  # Delete / Backspace
            if self.selected_indices or self.last_selected_idx is not None:
                self.delete_pressed = True

    # --- Decode scheduling --------------------------------------------------

    def _request_decodes(self, files, first, last):
        """Keep the worker pool focused on the current window.

        Cancel queued decodes that scrolled out of view, then submit the missing
        ones -- bounded and filling outward from the center -- so a fast scroll
        never queues a backlog for rows it flew past and the cursor's thumbnails
        always load first.
        """
        window = set(files[first:last])
        for path in [p for p in self._futures if p not in window]:
            if self._futures[path].cancel():   # only succeeds if it had not started
                del self._futures[path]

        if first >= last:
            return

        slots = MAX_IN_FLIGHT - len(self._futures)
        if slots <= 0:
            return
        center = (first + last) // 2
        todo = [i for i in range(first, last)
                if files[i] not in self._textures and files[i] not in self._futures]
        todo.sort(key=lambda i: abs(i - center))
        executor = self._get_executor()
        for i in todo[:slots]:
            path = files[i]
            self._futures[path] = executor.submit(self._decode, path, self._generation)

    def _cancel_all_futures(self):
        for future in self._futures.values():
            future.cancel()
        self._futures.clear()

    def _decode(self, path, generation):
        data = _render_thumbnail(path, self.thumbnail_size, self.padding_value)
        self._done.put((generation, path, data))

    def _get_executor(self):
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="thumb")
        return self._executor

    # --- Texture / cache housekeeping --------------------------------------

    def _make_texture(self, data):
        try:
            channels = data.shape[2] if data.ndim > 2 else 1
            return gl_utils.Texture(image=data, width=data.shape[1],
                                    height=data.shape[0], channels=channels)
        except Exception as e:
            logger.warning("Error creating thumbnail texture: %s", e)
            return None

    def _evict(self):
        while len(self._textures) > CACHE_LIMIT:
            _, texture = self._textures.popitem(last=False)  # oldest = least recently seen
            self._trash.append(texture)

    def _flush_trash(self):
        """Delete last frame's discarded textures, now that that frame has rendered."""
        for texture in self._trash:
            self._delete(texture)
        self._trash.clear()

    def _drain_done(self):
        while True:
            try:
                self._done.get_nowait()
            except queue.Empty:
                return

    @staticmethod
    def _delete(texture):
        if texture is not None and texture.gl_id is not None:
            try:
                texture.delete()
            except Exception:
                pass
