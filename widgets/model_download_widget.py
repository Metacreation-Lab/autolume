import glob
import math
import os
import threading
import time

import imgui

from utils import device_utils
from utils.downloads import download_file, load_catalog
from utils.gui_utils import imgui_utils

#----------------------------------------------------------------------------

class ModelDownloadWidget:
    """Catalog popup listing curated models plus a modal download progress popup."""

    def __init__(self, app, models_dir, on_complete=None):
        self.app = app
        self.models_dir = models_dir
        self.on_complete = on_complete
        self.catalog = load_catalog()
        # StyleGAN3 is unusably slow on the MPS backend, so hide those models on Apple Silicon.
        if device_utils.get_device().type == 'mps':
            self.catalog = [e for e in self.catalog if not e['architecture'].startswith('stylegan3')]
        self.show_browser = False
        # Dropdown that opened the catalog, auto-selected when its download finishes.
        self.requester = None

        self._state = 'idle'  # 'idle' | 'downloading' | 'error'
        self._bytes_done = 0
        self._total_bytes = 0
        self._error_msg = ''
        self._active_entry = None
        self._cancel_event = None
        self._thread = None
        self._finished_ok = False

        # Remove partial files left behind if the app was killed mid-download.
        for part in glob.glob(os.path.join(models_dir, '*.part')):
            try:
                os.remove(part)
            except OSError:
                pass

    def start_download(self, entry):
        if self._state != 'idle':
            return
        self._active_entry = entry
        self._bytes_done = 0
        self._total_bytes = 0
        self._error_msg = ''
        self._cancel_event = threading.Event()
        self._state = 'downloading'
        self._thread = threading.Thread(target=self._worker, args=(entry, self._cancel_event),
                                        name='ModelDownloadThread', daemon=True)
        self._thread.start()

    def _worker(self, entry, cancel_event):
        dest_path = os.path.join(self.models_dir, entry['filename'])

        def progress(done, total):
            self._bytes_done = done
            self._total_bytes = total

        try:
            ok = download_file(entry['url'], dest_path, cancel_event, progress)
        except Exception as e:
            self._error_msg = str(e)
            self._state = 'error'
            return
        if ok:
            self._finished_ok = True
        else:
            self._state = 'idle'

    @imgui_utils.scoped_by_object_id
    def __call__(self):
        # Consume the worker's completion flag on the UI thread.
        if self._finished_ok:
            self._finished_ok = False
            self._state = 'idle'
            completed_path = os.path.join(self.models_dir, self._active_entry['filename'])
            if self.requester is not None:
                self.requester.notify_downloaded(completed_path)
                self.requester = None
            if self.on_complete is not None:
                self.on_complete()

        if self.show_browser:
            imgui.open_popup('get_models_popup')
            self.show_browser = False

        self._draw_catalog_popup()
        self._draw_progress_modal()

    def _draw_catalog_popup(self):
        imgui.set_next_window_size(self.app.content_width // 2, 0)
        imgui.set_next_window_position(
            self.app.content_width * 0.5, self.app.content_height * 0.5,
            pivot_x=0.5, pivot_y=0.5)
        if imgui.begin_popup('get_models_popup'):
            if not self.catalog:
                imgui.text('Could not load model list')
            else:
                self._draw_catalog_rows()
            imgui.separator()
            imgui.spacing()
            if imgui_utils.button('Close', width=self.app.button_w):
                imgui.close_current_popup()
            imgui.end_popup()

    def _set_catalog_columns(self, name):
        cw = imgui.get_content_region_available_width()
        imgui.columns(6, name, border=False)
        imgui.set_column_width(0, int(cw * 0.26))
        imgui.set_column_width(1, int(cw * 0.11))
        imgui.set_column_width(2, int(cw * 0.13))
        imgui.set_column_width(3, int(cw * 0.15))
        imgui.set_column_width(4, int(cw * 0.22))

    def _draw_catalog_rows(self):
        self._set_catalog_columns('##model_catalog_header')
        for label in ('Name', 'Resolution', 'Architecture', 'Author', 'License', ''):
            imgui.text(label)
            imgui.next_column()
        imgui.columns(1)
        imgui.separator()

        self._set_catalog_columns('##model_catalog_rows')
        for entry in self.catalog:
            for col in ('name', 'resolution', 'architecture', 'author', 'license'):
                imgui.text(entry[col])
                imgui.next_column()
            downloaded = os.path.exists(os.path.join(self.models_dir, entry['filename']))
            if downloaded:
                imgui_utils.button(f"Downloaded##{entry['filename']}", width=-1, enabled=False)
            elif imgui_utils.button(f"Download##{entry['filename']}", width=-1, enabled=(self._state == 'idle')):
                self.start_download(entry)
                imgui.close_current_popup()
            imgui.next_column()
        imgui.columns(1)

    def _draw_progress_modal(self):
        popup_width = self.app.content_width // 2.5
        active = self._state in ('downloading', 'error')
        if active:
            imgui.open_popup('model_download_modal')
            imgui.set_next_window_position(self.app.content_width / 2 - popup_width / 2,
                                           self.app.content_height / 3)
            imgui.set_next_window_size(popup_width, 0)
        if imgui.begin_popup_modal('model_download_modal',
                                   flags=imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_MOVE)[0]:
            if self._state == 'downloading':
                imgui.text(f"Downloading {self._active_entry['name']}...")
                imgui.separator()
                imgui.spacing()
                done_mb = self._bytes_done / (1024 * 1024)
                if self._total_bytes > 0:
                    fraction = self._bytes_done / self._total_bytes
                    label = f'{fraction * 100:.1f}% ({done_mb:.1f} / {self._total_bytes / (1024 * 1024):.1f} MB)'
                else:
                    # Content length unknown: cycle the bar to show activity.
                    fraction = math.fmod(time.time(), 1.0)
                    label = f'{done_mb:.1f} MB downloaded'
                progress_width = popup_width - 40
                imgui.progress_bar(fraction, (progress_width, 20))
                text_width = imgui.calc_text_size(label)[0]
                imgui.set_cursor_pos_x((progress_width - text_width) / 2)
                imgui.text(label)
                imgui.spacing()
                if imgui_utils.button('Cancel', width=self.app.button_w):
                    self._cancel_event.set()
            elif self._state == 'error':
                imgui.text(f"Failed to download {self._active_entry['name']}")
                imgui.text_colored(self._error_msg, 1.0, 0.2, 0.2, 1.0)
                imgui.spacing()
                if imgui_utils.button('Close', width=self.app.button_w):
                    self._state = 'idle'
                    imgui.close_current_popup()
            else:
                # Download finished or was cancelled while the modal was open.
                imgui.close_current_popup()
            imgui.end_popup()

#----------------------------------------------------------------------------
