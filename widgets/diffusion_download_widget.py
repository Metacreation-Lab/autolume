import logging
import math
import os
import threading
import time

import imgui

from utils.diffusion_catalog import destination, is_installed, load_catalog
from utils.downloads import download_file
from utils.gui_utils import imgui_utils

logger = logging.getLogger(__name__)

# Civitai and HuggingFace both drop multi-gigabyte transfers, so a download gets
# several chances to resume before it is called a failure.
DOWNLOAD_TRIES = 6

#----------------------------------------------------------------------------

class DiffusionDownloadWidget:
    """Catalog popup and download modal for diffusion checkpoints.

    Same surface as ModelDownloadWidget so ModelDropdownButton can drive either
    one. Entries are single .safetensors files, as every other tool in this
    ecosystem uses, so a checkpoint is one download into the checkpoints folder.
    """

    def __init__(self, app, checkpoints_dir, on_complete=None):
        self.app = app
        self.checkpoints_dir = checkpoints_dir
        self.on_complete = on_complete
        self.catalog = load_catalog()
        self.show_browser = False
        self.requester = None

        self._state = 'idle'  # 'idle' | 'downloading' | 'error'
        self._bytes_done = 0
        self._total_bytes = 0
        self._error_msg = ''
        self._active_entry = None
        self._cancel_event = None
        self._thread = None
        self._finished_ok = False

        # a partial left by a killed app cannot be resumed across runs
        self._clear_partials()

    def _clear_partials(self):
        if not os.path.isdir(self.checkpoints_dir):
            return
        for name in os.listdir(self.checkpoints_dir):
            if name.endswith('.part'):
                try:
                    os.remove(os.path.join(self.checkpoints_dir, name))
                except OSError:
                    logger.warning('Could not remove stale partial %s', name)

    def start_download(self, entry):
        if self._state != 'idle':
            return
        self._active_entry = entry
        self._bytes_done = self._total_bytes = 0
        self._error_msg = ''
        self._cancel_event = threading.Event()
        self._state = 'downloading'
        self._thread = threading.Thread(target=self._worker, args=(entry, self._cancel_event),
                                        name='DiffusionDownloadThread', daemon=True)
        self._thread.start()

    def _worker(self, entry, cancel_event):
        def progress(done, total):
            self._bytes_done, self._total_bytes = done, total

        try:
            # download_file writes to a .part and renames on success, so the
            # checkpoint only ever appears once it is whole
            ok = download_file(entry['url'], destination(entry, self.checkpoints_dir),
                               cancel_event, progress, resume=True, tries=DOWNLOAD_TRIES)
        except Exception as e:
            logger.exception('Download of %s failed', entry['name'])
            self._error_msg = f'{type(e).__name__}: {e}'
            self._state = 'error'
            return
        if ok:
            self._finished_ok = True
        else:
            self._state = 'idle'

    @imgui_utils.scoped_by_object_id
    def __call__(self):
        if self._finished_ok:
            self._finished_ok = False
            self._state = 'idle'
            completed = destination(self._active_entry, self.checkpoints_dir)
            if self.requester is not None:
                self.requester.notify_downloaded(completed)
                self.requester = None
            if self.on_complete is not None:
                self.on_complete()

        if self.show_browser:
            imgui.open_popup('get_diffusion_models_popup')
            self.show_browser = False

        self._draw_catalog_popup()
        self._draw_progress_modal()

    def _set_columns(self, name):
        cw = imgui.get_content_region_available_width()
        imgui.columns(5, name, border=False)
        imgui.set_column_width(0, int(cw * 0.26))
        imgui.set_column_width(1, int(cw * 0.24))
        imgui.set_column_width(2, int(cw * 0.14))
        imgui.set_column_width(3, int(cw * 0.12))

    def _draw_catalog_popup(self):
        imgui.set_next_window_size(self.app.content_width // 2, 0)
        imgui.set_next_window_position(self.app.content_width * 0.5,
                                       self.app.content_height * 0.5,
                                       pivot_x=0.5, pivot_y=0.5)
        if imgui.begin_popup('get_diffusion_models_popup'):
            if not self.catalog:
                imgui.text('Could not load the model list')
            else:
                self._set_columns('##diffusion_catalog_header')
                for label in ('Name', 'Style', 'Base', 'Size', ''):
                    imgui.text(label)
                    imgui.next_column()
                imgui.columns(1)
                imgui.separator()

                self._set_columns('##diffusion_catalog_rows')
                for entry in self.catalog:
                    imgui.text(entry['name'])
                    imgui.next_column()
                    imgui.text(entry['style'])
                    imgui.next_column()
                    imgui.text(entry['base_model'])
                    imgui.next_column()
                    imgui.text(f"{int(entry['size_mb']) / 1024:.1f} GB")
                    imgui.next_column()
                    if is_installed(entry, self.checkpoints_dir):
                        imgui_utils.button(f"Downloaded##{entry['filename']}", width=-1, enabled=False)
                    elif imgui_utils.button(f"Download##{entry['filename']}", width=-1,
                                            enabled=(self._state == 'idle')):
                        self.start_download(entry)
                        imgui.close_current_popup()
                    imgui.next_column()
                imgui.columns(1)
            imgui.separator()
            imgui.spacing()
            if imgui_utils.button('Close', width=self.app.button_w):
                imgui.close_current_popup()
            imgui.end_popup()

    def _draw_progress_modal(self):
        width = self.app.content_width // 2.5
        if self._state in ('downloading', 'error'):
            imgui.open_popup('diffusion_download_modal')
            imgui.set_next_window_position(self.app.content_width / 2 - width / 2,
                                           self.app.content_height / 3)
            imgui.set_next_window_size(width, 0)
        if imgui.begin_popup_modal('diffusion_download_modal',
                                   flags=imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_MOVE)[0]:
            if self._state == 'downloading':
                imgui.text(f"Downloading {self._active_entry['name']}...")
                imgui.separator()
                imgui.spacing()
                done_mb = self._bytes_done / (1024 * 1024)
                if self._total_bytes > 0:
                    fraction = self._bytes_done / self._total_bytes
                    label = (f'{fraction * 100:.1f}% ({done_mb:.1f} / '
                             f'{self._total_bytes / (1024 * 1024):.1f} MB)')
                else:
                    fraction = math.fmod(time.time(), 1.0)
                    label = f'{done_mb:.1f} MB downloaded'
                bar_width = width - 40
                imgui.progress_bar(fraction, (bar_width, 20))
                imgui.set_cursor_pos_x((bar_width - imgui.calc_text_size(label)[0]) / 2)
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
                imgui.close_current_popup()
            imgui.end_popup()

#----------------------------------------------------------------------------
