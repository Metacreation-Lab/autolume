import os

import imgui

from utils.gui_utils import imgui_utils
from utils.model_dir import list_model_pkls, list_training_run_pkls

#----------------------------------------------------------------------------

class ModelDropdownButton:
    """Button opening a dropdown of local models and training run snapshots,
    optionally followed by a 'Download Models...' catalog entry.

    The model list is rescanned every time the dropdown opens, so models
    downloaded or saved elsewhere in the app show up without a restart.
    """

    def __init__(self, label='Models', items_provider=list_model_pkls,
                 runs_provider=list_training_run_pkls, include_models=True,
                 include_training_runs=True, show_download=False, downloader=None):
        if show_download:
            assert downloader is not None, 'show_download=True requires a downloader'
        self.downloader = downloader
        self.label = label
        self.items_provider = items_provider
        self.runs_provider = runs_provider
        self.include_models = include_models
        self.include_training_runs = include_training_runs
        self.show_download = show_download
        self.models = []
        self.run_items = []
        self._refocus = False
        self._pending_pick = None

    def notify_downloaded(self, pkl):
        """Queue a just-downloaded model to be selected on the next enabled frame."""
        self._pending_pick = pkl

    @imgui_utils.scoped_by_object_id
    def __call__(self, width=0, enabled=True):
        """Draw the button and dropdown; return the picked model path, or None."""
        picked = None
        if self._pending_pick is not None and enabled:
            picked = self._pending_pick
            self._pending_pick = None
        if imgui_utils.button(self.label, width=width, enabled=enabled):
            self.models = self.items_provider() if self.include_models else []
            self.run_items = self.runs_provider() if self.include_training_runs else []
            imgui.open_popup('model_dropdown')
            self._refocus = True
        if imgui.begin_popup('model_dropdown'):
            for pkl in self.models:
                clicked, _state = imgui.menu_item(f'{os.path.basename(pkl)}##{pkl}')
                if clicked:
                    picked = pkl
            if not self.models and not self.run_items:
                imgui.menu_item('No models found', None, False, False)
            if self._refocus:
                imgui.set_scroll_here()
                self._refocus = False
            if self.run_items:
                if self.models:
                    imgui.separator()
                for run, snapshots in self.run_items:
                    if imgui.begin_menu(run):
                        for name, pkl in snapshots:
                            clicked, _state = imgui.menu_item(f'{name}##{pkl}')
                            if clicked:
                                picked = pkl
                        imgui.end_menu()
            if self.show_download:
                imgui.separator()
                clicked, _state = imgui.menu_item('Download Models...')
                if clicked:
                    self.downloader.show_browser = True
                    self.downloader.requester = self
            imgui.end_popup()
        return picked

#----------------------------------------------------------------------------
