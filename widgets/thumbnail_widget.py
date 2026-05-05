import cv2
import os
import math
import collections
import queue as queue_mod
import multiprocessing as mp
import numpy as np
import imgui
from utils.gui_utils import gl_utils
import gc

from utils.dataset_preprocessing_utils import DatasetPreprocessingUtils


def _render_thumbnail(file_path, thumbnail_size):
    """Pure helper: render an image/video file into a square thumbnail np.array.

    Runs in worker processes. Returns None on failure.
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
    file_ext = os.path.splitext(file_path)[1].lower()

    try:
        if file_ext in video_extensions:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                return None
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return None
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elif len(frame.shape) == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
            img = frame
            padding_value = 0
        else:
            img = DatasetPreprocessingUtils().load_images(file_path)
            padding_value = 26

        if img is None:
            return None

        height, width = img.shape[:2]
        max_dim = max(height, width)
        channels = img.shape[2] if len(img.shape) > 2 else 1
        canvas = np.full((max_dim, max_dim, channels), padding_value, dtype=img.dtype)
        y_offset = (max_dim - height) // 2
        x_offset = (max_dim - width) // 2
        canvas[y_offset:y_offset + height, x_offset:x_offset + width] = img
        return cv2.resize(canvas, (thumbnail_size, thumbnail_size), interpolation=cv2.INTER_AREA)
    except Exception as e:
        print(f"Error rendering thumbnail for {file_path}: {e}")
        return None


class ThumbnailWidget:
    """Widget for handling image thumbnails with caching and display.

    Owns its own background worker process for rendered thumbnails. Each frame
    `render_thumbnails()` writes the latest desired set of paths to a single-slot
    request queue, and `poll()` consumes streamed results. A generation counter
    invalidates stale results when the file list or render mode changes.
    """

    def __init__(self):
        self.thumbnail_size = 140  # Determines quality of thumbnails (resolution)
        self.thumbnails = {}  
        self.placeholder_textures = {}  
        self.generate_thumbnails = False  # Show rendered thumbnails (when available) vs placeholders
        self.selected_files = []
        self.last_selected_idx = None
        self.selected_indices = []
        self.delete_pressed = False

        self._deferred_delete = []
        self._frame_held_textures = []

        # FIFO cache for rendered thumbnails outside the visible+buffer window.
        self._cached_thumbnails = collections.OrderedDict()
        self.max_cached_thumbnails = 1000

        self._req_q = None       # mp.Queue(maxsize=1) — latest-wins payload
        self._rep_q = None       # mp.Queue() — streamed results
        self._worker_proc = None
        self._generation = 0     # bumped on file-list / render-mode change
        self._render_mode_active = False  # tracks worker-intent independently from the public flag

    def create_placeholder_thumbnail(self, file_path):
        """Create a grey placeholder thumbnail with image name"""
        try:
            size = self.thumbnail_size
            canvas = np.full((size, size, 3), [128, 128, 128], dtype=np.uint8)

            filename = os.path.splitext(os.path.basename(file_path))[0]

            canvas = self._add_text_to_canvas(canvas, filename, size)

            # Create texture
            texture = gl_utils.Texture(
                image=canvas,
                width=size,
                height=size,
                channels=3
            )

            return texture
        except Exception as e:
            print(f"Failed to create placeholder thumbnail for {file_path}: {e}")
            return None

    def _add_text_to_canvas(self, canvas, text, size):
        """Add text to the center of the thumbnail placeholder"""
        try:
            font_scale = 0.4
            font_thickness = 1
            font = cv2.FONT_HERSHEY_SIMPLEX

            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

            max_width = int(size * 0.9)
            if text_width > max_width:
                char_width = text_width / len(text)
                max_chars = int(max_width / char_width) - 3
                text = text[:max_chars] + "..."
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

            x = (size - text_width) // 2
            y = (size + text_height) // 2

            cv2.putText(canvas, text, (x, y), font, font_scale, (255, 255, 255), font_thickness)

            return canvas
        except Exception as e:
            print(f"Failed to add text to canvas: {e}")
            return canvas

    def get_thumbnail(self, file_path):
        """Return rendered texture if available + render mode on, else placeholder.

        Promotes a cached texture back into the active set when the path
        re-enters the render window so subsequent demotions land it at the
        tail of the FIFO again.
        """
        if self.generate_thumbnails:
            if file_path in self.thumbnails:
                return self.thumbnails[file_path]
            cached = self._cached_thumbnails.pop(file_path, None)
            if cached is not None:
                self.thumbnails[file_path] = cached
                return cached
        if file_path not in self.placeholder_textures:
            self.placeholder_textures[file_path] = self.create_placeholder_thumbnail(file_path)
        return self.placeholder_textures[file_path]

    # --- Render mode + worker lifecycle -------------------------------------

    def set_render_mode(self, enabled):
        """Toggle whether the widget displays rendered thumbnails and feeds the worker.

        Lazy-starts the worker on first enable. Disabling does NOT stop the worker
        (so re-enabling is instant) and does NOT clear `self.thumbnails`.
        Idempotent: only bumps generation on real state changes so callers can
        invoke this every frame without dropping in-flight results.
        """
        enabled = bool(enabled)
        self.generate_thumbnails = enabled
        if enabled != self._render_mode_active:
            self._render_mode_active = enabled
            self._generation += 1
        if enabled:
            self._ensure_worker()

    def _ensure_worker(self):
        """Start the worker process and queues if not already running."""
        if self._worker_proc is not None and self._worker_proc.is_alive():
            return
        self._req_q = mp.Queue(maxsize=1)
        self._rep_q = mp.Queue()
        self._worker_proc = mp.Process(
            target=ThumbnailWidget._worker_main,
            args=(self._req_q, self._rep_q),
            daemon=True,
        )
        self._worker_proc.start()

    def poll(self):
        """Drain reply queue and apply results matching the current generation."""
        if self._rep_q is None:
            return
        try:
            while True:
                msg = self._rep_q.get_nowait()
                if msg.get('generation') != self._generation:
                    continue
                file_path = msg.get('file_path')
                data = msg.get('data')
                if file_path is None:
                    continue
                self.update_thumbnail_from_data(file_path, data)
        except queue_mod.Empty:
            return
        except Exception as e:
            print(f"Error polling thumbnail results: {e}")

    def shutdown(self):
        """Tear down the worker process and queues without raising BrokenPipeError."""
        if self._req_q is not None:
            try:
                try:
                    self._req_q.get_nowait()
                except Exception:
                    pass
                try:
                    self._req_q.put_nowait(None)
                except Exception:
                    pass
            except Exception:
                pass

        proc = self._worker_proc
        if proc is not None:
            try:
                proc.join(timeout=0.5)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=0.5)
                if proc.is_alive():
                    proc.kill()
                    proc.join()
            except Exception as e:
                print(f"Error stopping thumbnail worker: {e}")

        for q in (self._req_q, self._rep_q):
            if q is None:
                continue
            try:
                q.cancel_join_thread()
                q.close()
            except Exception:
                pass

        self._worker_proc = None
        self._req_q = None
        self._rep_q = None

    @staticmethod
    def _worker_main(req_q, rep_q):
        """Worker entry point: latest-wins payload, preempts between images."""
        current = None
        idx = 0
        while True:
            payload = None
            got_payload = False
            try:
                if current is None:
                    payload = req_q.get(timeout=0.25)
                else:
                    payload = req_q.get_nowait()
                got_payload = True
            except queue_mod.Empty:
                got_payload = False
            except Exception:
                return

            if got_payload:
                if payload is None:
                    return 
                if isinstance(payload, dict):
                    current = payload
                    idx = 0
                else:
                    continue

            if current is None:
                continue

            paths = current.get('paths') or []
            size = current.get('size', 140)
            gen_id = current.get('generation', 0)

            if idx >= len(paths):
                current = None
                continue

            path = paths[idx]
            idx += 1
            data = _render_thumbnail(path, size)
            try:
                rep_q.put({'generation': gen_id, 'file_path': path, 'data': data})
            except Exception:
                return

    # --- File list management -----------------------------------------------

    def set_thumbnail_mode(self, generate_thumbnails, prev_thumbnail_mode=None):
        """Public toggle: route to set_render_mode and clear placeholders.

        Rendered cache survives the toggle so re-enabling is instant.
        """
        prev = prev_thumbnail_mode if prev_thumbnail_mode is not None else self.generate_thumbnails
        if prev == generate_thumbnails:
            return
        self.clear_placeholder_thumbnails()
        self.set_render_mode(generate_thumbnails)

    def update_thumbnails(self, file_paths):
        """Update the file list. Textures are created lazily during render."""
        self.selected_files = file_paths
        self._generation += 1
        valid = set(file_paths)
        stale = [fp for fp in self._cached_thumbnails if fp not in valid]
        for fp in stale:
            tex = self._cached_thumbnails.pop(fp)
            if tex is not None:
                self._deferred_delete.append(tex)

    def clear_thumbnails(self):
        """Defer-delete all rendered thumbnails."""
        for tex in self.thumbnails.values():
            if tex is not None:
                self._deferred_delete.append(tex)
        self.thumbnails.clear()

    def clear_placeholder_thumbnails(self):
        """Defer-delete all placeholder thumbnails."""
        for tex in self.placeholder_textures.values():
            if tex is not None:
                self._deferred_delete.append(tex)
        self.placeholder_textures.clear()

    def clear_cached_thumbnails(self):
        """Defer-delete all FIFO-cached offscreen thumbnails."""
        for tex in self._cached_thumbnails.values():
            if tex is not None:
                self._deferred_delete.append(tex)
        self._cached_thumbnails.clear()

    def clear_all_thumbnails(self):
        """Clear actual, cached, and placeholder thumbnails"""
        self.clear_thumbnails()
        self.clear_cached_thumbnails()
        self.clear_placeholder_thumbnails()

    def _flush_deferred_deletes(self):
        """Delete textures scheduled for deferred deletion (safe after prior frame rendered)."""
        for tex in self._deferred_delete:
            if tex is not None:
                try:
                    tex.delete()
                except Exception:
                    pass
        self._deferred_delete.clear()

    def _cleanup_offscreen_placeholders(self, visible_files):
        """Free placeholder textures that are outside the visible + buffer range."""
        visible_set = set(visible_files)
        to_remove = [fp for fp in self.placeholder_textures if fp not in visible_set]
        for fp in to_remove:
            tex = self.placeholder_textures.pop(fp)
            if tex is not None:
                self._deferred_delete.append(tex)

    def _demote_offscreen_thumbnails(self, visible_files):
        """Move rendered thumbnails outside the render window into the FIFO cache,
        evicting the oldest cached entries once the limit is exceeded."""
        visible_set = set(visible_files)
        to_demote = [fp for fp in self.thumbnails if fp not in visible_set]
        for fp in to_demote:
            tex = self.thumbnails.pop(fp)
            if tex is None:
                continue

            if fp in self._cached_thumbnails:
                old = self._cached_thumbnails.pop(fp)
                if old is not None and old is not tex:
                    self._deferred_delete.append(old)
            self._cached_thumbnails[fp] = tex

        while len(self._cached_thumbnails) > self.max_cached_thumbnails:
            _, evicted = self._cached_thumbnails.popitem(last=False)
            if evicted is not None:
                self._deferred_delete.append(evicted)

    # --- Rendering ----------------------------------------------------------

    def render_thumbnails(self, available_width, available_height):
        self._flush_deferred_deletes()
        self._frame_held_textures = []

        if not self.selected_files:
            message = "No images imported"
            text_width = imgui.calc_text_size(message)[0]
            text_height = imgui.get_text_line_height()

            center_x = (available_width - text_width) / 2
            center_y = (available_height - text_height) / 2

            imgui.set_cursor_pos_x(center_x)
            imgui.set_cursor_pos_y(center_y)

            imgui.text_colored(message, 0.5, 0.5, 0.5, 1.0)
            return

        min_thumb_size = 120
        max_thumb_size = 220
        spacing_x = 32
        spacing_y = 16
        grid_padding = 6
        n = len(self.selected_files)

        inner_width = available_width - 2 * grid_padding

        thumbnails_per_row = max(1, int((inner_width + spacing_x) // (min_thumb_size + spacing_x)))
        thumb_size = min(
            max_thumb_size,
            max(min_thumb_size, int((inner_width - (thumbnails_per_row - 1) * spacing_x) // thumbnails_per_row))
        )

        total_row_width = thumbnails_per_row * thumb_size + (thumbnails_per_row - 1) * spacing_x
        left_margin = grid_padding + max(0, (inner_width - total_row_width) // 2)

        text_line_height = imgui.get_text_line_height_with_spacing()
        row_height = thumb_size + spacing_y + text_line_height
        total_rows = math.ceil(n / thumbnails_per_row)

        scroll_y = imgui.get_scroll_y()
        first_visible_row = max(0, int(scroll_y / row_height))
        last_visible_row = int((scroll_y + available_height) / row_height)

        thumbnail_buffer = 100
        buffer_rows = math.ceil(thumbnail_buffer / thumbnails_per_row)
        first_render_row = max(0, first_visible_row - buffer_rows)
        last_render_row = min(total_rows - 1, last_visible_row + buffer_rows)

        first_render_idx = first_render_row * thumbnails_per_row
        last_render_idx = min(n, (last_render_row + 1) * thumbnails_per_row)

        if first_render_row > 0:
            imgui.dummy(available_width, first_render_row * row_height)

        ctrl_down = imgui.is_key_down(341) or imgui.is_key_down(345)
        shift_down = imgui.is_key_down(340) or imgui.is_key_down(344)
        a_pressed = imgui.is_key_pressed(65)
        delete_pressed = imgui.is_key_pressed(261) or imgui.is_key_pressed(259)

        if ctrl_down and a_pressed:
            self.select_all()

        if delete_pressed and (self.selected_indices or self.last_selected_idx is not None):
            self.delete_pressed = True

        rendered_files = []

        for idx in range(first_render_idx, last_render_idx):
            file_path = self.selected_files[idx]
            texture = self.get_thumbnail(file_path)
            rendered_files.append(file_path)

            if texture is not None and hasattr(texture, 'gl_id') and texture.gl_id is not None:
                col = idx % thumbnails_per_row

                if col == 0:
                    imgui.dummy(left_margin, 0)
                    imgui.same_line(spacing=0)
                elif col > 0:
                    imgui.same_line(spacing=spacing_x)

                imgui.begin_group()
                imgui.push_id(str(idx))
                is_hovered = False

                if self.selected_indices:
                    is_selected = (idx in self.selected_indices)
                else:
                    is_selected = (self.last_selected_idx == idx)

                cursor_pos = imgui.get_cursor_screen_pos()
                draw_list = imgui.get_window_draw_list()
                frame_color = (0.8, 0.4, 0.1, 1.0) if is_selected else (1.0, 0.8, 0.2, 0.7)
                border_thickness = 3 if is_selected else 2

                if imgui.invisible_button("thumb", thumb_size, thumb_size):
                    if shift_down and self.last_selected_idx is not None:
                        start = min(self.last_selected_idx, idx)
                        end = max(self.last_selected_idx, idx)
                        self.selected_indices = list(range(start, end + 1))
                    elif ctrl_down:
                        if idx in self.selected_indices:
                            self.selected_indices.remove(idx)
                        else:
                            self.selected_indices.append(idx)
                        self.last_selected_idx = idx
                    else:
                        self.selected_indices = [idx]
                        self.last_selected_idx = idx

                is_hovered = imgui.is_item_hovered()

                imgui.set_cursor_screen_pos((cursor_pos[0], cursor_pos[1]))
                self._frame_held_textures.append(texture)
                imgui.image(int(texture.gl_id), thumb_size, thumb_size)

                if is_hovered or is_selected:
                    x1, y1 = cursor_pos
                    x2, y2 = x1 + thumb_size, y1 + thumb_size
                    draw_list.add_rect(x1, y1, x2, y2, imgui.get_color_u32_rgba(*frame_color), rounding=6, thickness=border_thickness)

                filename = os.path.basename(file_path)
                if len(filename) > 15:
                    filename = filename[:12] + "..."
                text_width = imgui.calc_text_size(filename)[0]
                cursor_x = imgui.get_cursor_pos_x()
                imgui.set_cursor_pos_x(cursor_x + (thumb_size - text_width) / 2)
                imgui.text(filename)
                imgui.pop_id()
                imgui.end_group()

                if col == thumbnails_per_row - 1:
                    imgui.new_line()
                    imgui.dummy(0, spacing_y)

        remaining_rows = total_rows - 1 - last_render_row
        if remaining_rows > 0:
            imgui.dummy(available_width, remaining_rows * row_height)

        self._cleanup_offscreen_placeholders(rendered_files)
        self._demote_offscreen_thumbnails(rendered_files)

        # Schedule worker to render the visible/buffer set, viewport-first.
        if self.generate_thumbnails:
            self._schedule_render_requests(
                first_render_idx, last_render_idx,
                first_visible_row, last_visible_row,
                thumbnails_per_row,
            )

    def _schedule_render_requests(self, first_render_idx, last_render_idx,
                                  first_visible_row, last_visible_row,
                                  thumbnails_per_row):
        """Build a center-out list of needed paths and post it to the worker slot."""
        if self._req_q is None:
            self._ensure_worker()
            if self._req_q is None:
                return

        visible_first_idx = max(first_render_idx, first_visible_row * thumbnails_per_row)
        visible_last_idx = min(last_render_idx, (last_visible_row + 1) * thumbnails_per_row)

        ordered_indices = list(range(visible_first_idx, visible_last_idx))
        above = list(range(visible_first_idx - 1, first_render_idx - 1, -1))
        below = list(range(visible_last_idx, last_render_idx))
        a = b = 0
        while a < len(above) or b < len(below):
            if b < len(below):
                ordered_indices.append(below[b])
                b += 1
            if a < len(above):
                ordered_indices.append(above[a])
                a += 1

        needed = [self.selected_files[i]
                  for i in ordered_indices
                  if self.selected_files[i] not in self.thumbnails]
        if not needed:
            return

        payload = {
            'generation': self._generation,
            'paths': needed,
            'size': self.thumbnail_size,
        }
        # Latest-wins: drain any stale slot, then put the new payload.
        try:
            self._req_q.get_nowait()
        except Exception:
            pass
        try:
            self._req_q.put_nowait(payload)
        except Exception:
            pass

    # --- Selection / utilities ---------------------------------------------

    def get_thumbnail_count(self):
        """Get the number of thumbnails"""
        return len(self.selected_files)

    def get_selected_files(self):
        """Get the list of selected files"""
        return self.selected_files.copy()

    def select_all(self):
        """Select all images"""
        self.selected_indices = list(range(len(self.selected_files)))
        self.last_selected_idx = None

    def get_selected_indices(self):
        if self.selected_indices:
            return sorted(set(self.selected_indices))
        elif self.last_selected_idx is not None:
            return [self.last_selected_idx]
        return []

    def clear_selected(self):
        self.last_selected_idx = None
        self.selected_indices = []

    def is_delete_pressed(self):
        """Check if delete was pressed and reset the flag"""
        if self.delete_pressed:
            self.delete_pressed = False
            return True
        return False

    def update_thumbnail_from_data(self, file_path, thumbnail_data):
        """Install a worker result as a rendered texture, deferring old texture deletion."""
        if thumbnail_data is None:
            return
        texture = self._create_texture_from_data(thumbnail_data)
        if texture is None:
            return
        old = self.thumbnails.get(file_path)
        if old is not None:
            self._deferred_delete.append(old)
        self.thumbnails[file_path] = texture

    def _create_texture_from_data(self, thumbnail_data):
        """Create OpenGL texture from thumbnail data"""
        try:
            return gl_utils.Texture(
                image=thumbnail_data,
                width=thumbnail_data.shape[1],
                height=thumbnail_data.shape[0],
                channels=thumbnail_data.shape[2] if len(thumbnail_data.shape) > 2 else 1
            )
        except Exception as e:
            print(f"Error creating texture from thumbnail data: {e}")
            return None

    # --- Cleanup ------------------------------------------------------------

    def cleanup(self):
        """Tear down worker and free all textures."""
        try:
            self.shutdown()
            self.generate_thumbnails = False
            self._flush_deferred_deletes()
            self._frame_held_textures = []

            for texture in self.thumbnails.values():
                if texture is not None and hasattr(texture, 'delete') and callable(texture.delete):
                    try:
                        if texture.gl_id is not None:
                            texture.delete()
                    except Exception as e:
                        print(f"Error deleting thumbnail texture: {e}")
            self.thumbnails.clear()

            for texture in self._cached_thumbnails.values():
                if texture is not None and hasattr(texture, 'delete') and callable(texture.delete):
                    try:
                        if texture.gl_id is not None:
                            texture.delete()
                    except Exception as e:
                        print(f"Error deleting cached thumbnail texture: {e}")
            self._cached_thumbnails.clear()

            for texture in self.placeholder_textures.values():
                if texture is not None and hasattr(texture, 'delete') and callable(texture.delete):
                    try:
                        if texture.gl_id is not None:
                            texture.delete()
                    except Exception as e:
                        print(f"Error deleting placeholder texture: {e}")
            self.placeholder_textures.clear()

            self.selected_files = []
            self.selected_indices = []
            self.last_selected_idx = None

            gc.collect()

        except Exception as e:
            print(f"Warning: Error during thumbnail widget cleanup: {e}")

    def __del__(self):
        pass
