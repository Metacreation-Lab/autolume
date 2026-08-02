# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import logging
import os
import re

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

import dnnlib
import imgui
from utils.gui_utils import imgui_utils
from utils.model_dir import list_model_pkls, models_dir, resolve_pkl
from widgets.model_download_widget import ModelDownloadWidget
from widgets.model_dropdown_widget import ModelDropdownButton
from widgets.model_input_widget import ModelInputWidget

from . import renderer

logger = logging.getLogger(__name__)

#----------------------------------------------------------------------------

class PickleWidget:
    def __init__(self, viz):
        self.viz            = viz
        self.cur_pkl        = None
        self.user_pkl       = ''
        self.recent_pkls    = []
        self.browse_cache = []
        self.load_pkl('', ignore_errors=True)
        self.use_osc = False
        self.osc_addresses = ""

        self.rescan_models()
        self.model_downloader = ModelDownloadWidget(viz.app, models_dir=models_dir(), on_complete=self.rescan_models)
        self.model_dropdown = ModelDropdownButton(show_download=True, downloader=self.model_downloader)
        self.model_input = ModelInputWidget(viz.app, dropdown=self.model_dropdown)

    def rescan_models(self):
        self.browse_cache = list_model_pkls()

    def get_params(self):
        return (self.recent_pkls, self.browse_cache, self.cur_pkl, self.user_pkl, self.use_osc, self.osc_addresses)

    def set_params(self, params):
        self.recent_pkls, self.browse_cache, self.cur_pkl, self.user_pkl, self.use_osc, self.osc_addresses = params

    def add_recent(self, pkl, ignore_errors=False):
        try:
            resolved = resolve_pkl(pkl)
            if resolved not in self.recent_pkls:
                self.recent_pkls.append(resolved)
        except:
            if not ignore_errors:
                raise

    def load_pkl(self, pkl, ignore_errors=False):
        viz = self.viz
        viz.app.skip_frame() # The input field will change on next frame.
        try:
            resolved = resolve_pkl(pkl)
            name = resolved.replace('\\', '/').split('/')[-1]
            self.cur_pkl = resolved
            self.user_pkl = resolved
            viz.result.message = f'Loading {name}...'
            viz.defer_rendering()
            if resolved in self.recent_pkls:
                self.recent_pkls.remove(resolved)
            self.recent_pkls.insert(0, resolved)
        except:
            self.cur_pkl = None
            self.user_pkl = pkl
            if pkl == '':
                viz.result = dnnlib.EasyDict(message='No network pickle loaded')
            else:
                viz.result = dnnlib.EasyDict(error=renderer.CapturedException())
            if not ignore_errors:
                raise

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.get_params(), f)

    def load(self, path):
        with open(path, "rb") as f:
            self.set_params(pickle.load(f))
        if not os.path.exists(self.user_pkl):
            head, tail = os.path.split(self.user_pkl)
            if os.path.exists(os.path.join(models_dir(), tail)):
                self.user_pkl = os.path.join(models_dir(), tail)
            else:
                logger.error("Model does not exist in the model folder")
        if not os.path.exists(self.cur_pkl):
            head, tail = os.path.split(self.cur_pkl)
            if os.path.exists(os.path.join(models_dir(), tail)):
                self.cur_pkl = os.path.join(models_dir(), tail)
            else:
                logger.error("Model does not exist in the model folder")
        for i, recent_pkl in enumerate(self.recent_pkls):
            if not os.path.exists(recent_pkl):
                head, tail = os.path.split(recent_pkl)
                candidate = os.path.join(models_dir(), tail)
                if os.path.exists(candidate):
                    self.recent_pkls[i] = candidate
                else:
                    logger.error("Model does not exist in the model folder")

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            imgui.text('Pickle')
            imgui.same_line(viz.app.label_w)
            changed, self.user_pkl = self.model_input(self.user_pkl, width=-1)
            if changed:
                self.load_pkl(self.user_pkl, ignore_errors=True)

        self.model_downloader()

        paths = viz.app.pop_drag_and_drop_paths()
        if paths is not None and len(paths) >= 1:
            self.load_pkl(paths[0], ignore_errors=True)

        viz.args.pkl = self.cur_pkl

    def list_runs_and_pkls(self, parents):
        items = []
        run_regex = re.compile(r'\d+-.*')
        pkl_regex = re.compile(r'network-snapshot-\d+\.pkl')
        for parent in set(parents):
            if os.path.isdir(parent):
                for entry in os.scandir(parent):
                    if entry.is_dir() and run_regex.fullmatch(entry.name):
                        items.append(dnnlib.EasyDict(type='run', name=entry.name, path=os.path.join(parent, entry.name)))
                    if entry.is_file() and pkl_regex.fullmatch(entry.name):
                        items.append(dnnlib.EasyDict(type='pkl', name=entry.name, path=os.path.join(parent, entry.name)))

        items = sorted(items, key=lambda item: (item.name.replace('_', ' '), item.path))
        return items

#----------------------------------------------------------------------------
