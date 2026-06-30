import csv
import glob
import html
import math
import os
import re
import threading
import time

import imgui
import requests

from utils.gui_utils import imgui_utils
from utils.model_dir import list_model_pkls
from utils.resource_paths import resource_path

#----------------------------------------------------------------------------
# Download core. GUI-free so it can be exercised headless.

CHUNK_SIZE = 1024 * 1024
REQUIRED_COLUMNS = ('name', 'filename', 'resolution', 'author', 'license', 'url')


def load_catalog():
    """Parse the bundled models.csv into a list of catalog entries."""
    csv_path = resource_path('models.csv')
    try:
        with open(csv_path, newline='', encoding='utf-8') as f:
            rows = []
            for row in csv.DictReader(f):
                if all(row.get(col) for col in REQUIRED_COLUMNS):
                    rows.append({col: row[col].strip() for col in REQUIRED_COLUMNS})
                else:
                    print(f'Skipping malformed models.csv row: {row}')
            return rows
    except OSError as e:
        print(f'Could not load models.csv: {e}')
        return []


def _resolve_google_drive(session, response):
    """Follow the Google Drive interstitial page to the actual file response."""
    page = response.text
    if 'Google Drive - Quota exceeded' in page:
        raise IOError('Google Drive download quota exceeded -- please try again later')
    # Modern interstitial: a form pointing at drive.usercontent.google.com with hidden params.
    match = re.search(r'<form[^>]*id="download-form"[^>]*action="([^"]+)"', page)
    if match:
        action = html.unescape(match.group(1))
        params = {name: html.unescape(value) for name, value in
                  re.findall(r'<input type="hidden" name="([^"]+)" value="([^"]*)"', page)}
        return session.get(action, params=params, stream=True, timeout=(10, 30))
    # Legacy interstitial: scrape the export=download confirmation link.
    links = [html.unescape(link) for link in page.split('"') if 'export=download' in link]
    if len(links) == 1:
        return session.get(requests.compat.urljoin(response.url, links[0]), stream=True, timeout=(10, 30))
    raise IOError('Could not resolve Google Drive download link')


def download_file(url, dest_path, cancel_event, progress_cb):
    """Stream url into dest_path. Returns False if cancelled, raises on failure.

    Data is written to dest_path + '.part' and atomically renamed on success,
    so dest_path only ever exists as a complete file.
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    part_path = dest_path + '.part'
    try:
        with requests.Session() as session:
            response = session.get(url, stream=True, timeout=(10, 30))
            response.raise_for_status()
            if response.headers.get('Content-Type', '').startswith('text/html'):
                response = _resolve_google_drive(session, response)
                response.raise_for_status()
                if response.headers.get('Content-Type', '').startswith('text/html'):
                    raise IOError('Could not resolve download link (quota exceeded or page layout changed)')
            total = int(response.headers.get('Content-Length', 0))
            done = 0
            with open(part_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if cancel_event.is_set():
                        return False
                    f.write(chunk)
                    done += len(chunk)
                    progress_cb(done, total)
        os.replace(part_path, dest_path)
        return True
    finally:
        if os.path.exists(part_path):
            try:
                os.remove(part_path)
            except OSError:
                pass

#----------------------------------------------------------------------------

class ModelDownloadWidget:
    """Catalog popup listing curated models plus a modal download progress popup."""

    def __init__(self, app, models_dir, on_complete=None):
        self.app = app
        self.models_dir = models_dir
        self.on_complete = on_complete
        self.catalog = load_catalog()
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
        if imgui.begin_popup('get_models_popup'):
            if not self.catalog:
                imgui.text('Could not load model list')
            else:
                self._draw_catalog_rows()
            imgui.end_popup()

    def _set_catalog_columns(self, name):
        cw = imgui.get_content_region_available_width()
        imgui.columns(5, name, border=False)
        imgui.set_column_width(0, int(cw * 0.30))
        imgui.set_column_width(1, int(cw * 0.12))
        imgui.set_column_width(2, int(cw * 0.18))
        imgui.set_column_width(3, int(cw * 0.25))

    def _draw_catalog_rows(self):
        self._set_catalog_columns('##model_catalog_header')
        for label in ('Name', 'Resolution', 'Author', 'License', ''):
            imgui.text(label)
            imgui.next_column()
        imgui.columns(1)
        imgui.separator()

        self._set_catalog_columns('##model_catalog_rows')
        for entry in self.catalog:
            for col in ('name', 'resolution', 'author', 'license'):
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

class ModelDropdownButton:
    """Button opening a dropdown of local models plus a 'Get Models...' catalog entry.

    The model list is rescanned every time the dropdown opens, so models
    downloaded or saved elsewhere in the app show up without a restart.
    """

    def __init__(self, downloader, label='Models', items_provider=list_model_pkls):
        self.downloader = downloader
        self.label = label
        self.items_provider = items_provider
        self.models = []
        self._refocus = False
        self._pending_pick = None

    def notify_downloaded(self, pkl):
        """Queue a just-downloaded model to be selected on the next frame."""
        self._pending_pick = pkl

    @imgui_utils.scoped_by_object_id
    def __call__(self, width=0):
        """Draw the button and dropdown; return the picked model path, or None."""
        picked = None
        if self._pending_pick is not None:
            picked = self._pending_pick
            self._pending_pick = None
        if imgui_utils.button(self.label, width=width):
            self.models = self.items_provider()
            imgui.open_popup('model_dropdown')
            self._refocus = True
        if imgui.begin_popup('model_dropdown'):
            for pkl in self.models:
                clicked, _state = imgui.menu_item(f'{os.path.basename(pkl)}##{pkl}')
                if clicked:
                    picked = pkl
            if not self.models:
                imgui.menu_item('No models found', None, False, False)
            if self._refocus:
                imgui.set_scroll_here()
                self._refocus = False
            imgui.separator()
            clicked, _state = imgui.menu_item('Get Models...')
            if clicked:
                self.downloader.show_browser = True
                self.downloader.requester = self
            imgui.end_popup()
        return picked

#----------------------------------------------------------------------------
