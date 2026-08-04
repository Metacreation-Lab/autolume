import logging
import math
import os
import shutil
import threading
import time

import imgui

from utils.diffusion_catalog import is_installed, load_catalog, resolve_files
from utils.downloads import download_file
from utils.gui_utils import imgui_utils

logger = logging.getLogger(__name__)

# Both hosts drop multi-gigabyte transfers, so every file gets several chances
# to resume before the entry is called a failure.
TRIES_PER_FILE = 6

#----------------------------------------------------------------------------

class DiffusionDownloadWidget:
    """Catalog popup and download modal for diffusion checkpoints.

    Same surface as ModelDownloadWidget so ModelDropdownButton can drive either
    one, but an entry here is a list of files rather than a single URL: most of
    the catalog is HuggingFace repos, which are directories.
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
        self._file_index = 0
        self._file_count = 0
        self._error_msg = ''
        self._active_entry = None
        self._cancel_event = None
        self._thread = None
        self._finished_ok = False

        # A staging folder left by a killed app is not resumable across runs:
        # the file list may have changed, so it is cheaper to start over.
        self._clear_staging()

    def _staging_path(self, entry):
        return os.path.join(self.checkpoints_dir, entry['dest'] + '.part')

    def _clear_staging(self):
        if not os.path.isdir(self.checkpoints_dir):
            return
        for name in os.listdir(self.checkpoints_dir):
            if name.endswith('.part'):
                path = os.path.join(self.checkpoints_dir, name)
                try:
                    shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
                except OSError:
                    logger.warning('Could not remove stale staging path %s', path)

    def start_download(self, entry):
        if self._state != 'idle':
            return
        self._active_entry = entry
        self._bytes_done = self._total_bytes = 0
        self._file_index = self._file_count = 0
        self._error_msg = ''
        self._cancel_event = threading.Event()
        self._state = 'downloading'
        self._thread = threading.Thread(target=self._worker, args=(entry, self._cancel_event),
                                        name='DiffusionDownloadThread', daemon=True)
        self._thread.start()

    def _worker(self, entry, cancel_event):
        staging = self._staging_path(entry)
        try:
            files = resolve_files(entry)
            self._file_count = len(files)
            for index, (url, relative) in enumerate(files):
                if cancel_event.is_set():
                    break
                self._file_index = index + 1
                self._bytes_done = self._total_bytes = 0
                dest = os.path.join(staging, relative) if entry['source'] == 'hf' else staging

                def progress(done, total):
                    self._bytes_done, self._total_bytes = done, total

                if not download_file(url, dest, cancel_event, progress,
                                     resume=True, tries=TRIES_PER_FILE):
                    break  # cancelled
            else:
                # Only now does the folder become visible under its real name.
                # A half-downloaded diffusers folder has an index and no
                # weights, which reads as installed and refuses to load.
                final = os.path.join(self.checkpoints_dir, entry['dest'])
                if os.path.exists(final):
                    shutil.rmtree(final) if os.path.isdir(final) else os.remove(final)
                os.replace(staging, final)
                self._finished_ok = True
                return
        except Exception as e:
            logger.exception('Download of %s failed', entry['name'])
            self._error_msg = f'{type(e).__name__}: {e}'
            self._state = 'error'
            self._remove(staging)
            return
        self._remove(staging)
        self._state = 'idle'

    @staticmethod
    def _remove(path):
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            elif os.path.exists(path):
                os.remove(path)
        except OSError:
            logger.warning('Could not remove %s', path)

    @imgui_utils.scoped_by_object_id
    def __call__(self):
        if self._finished_ok:
            self._finished_ok = False
            self._state = 'idle'
            completed = os.path.join(self.checkpoints_dir, self._active_entry['dest'])
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
                        imgui_utils.button(f"Downloaded##{entry['dest']}", width=-1, enabled=False)
                    elif imgui_utils.button(f"Download##{entry['dest']}", width=-1,
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
                if self._file_count:
                    imgui.text(f'File {self._file_index} of {self._file_count}')
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
