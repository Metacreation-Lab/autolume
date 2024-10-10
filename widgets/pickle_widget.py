# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import glob
import os
import re

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

import dnnlib
import imgui
import numpy as np
from utils.gui_utils import imgui_utils
from widgets import browse_widget

from . import renderer

#----------------------------------------------------------------------------

def _locate_results(pattern):
    return pattern

#----------------------------------------------------------------------------

class PickleWidget:
    def __init__(self, viz):
        self.viz            = viz
        self.cur_pkl        = None
        self.user_pkl       = ''
        self.recent_pkls    = []
        self.browse_cache = []
        self.browse_refocus = False
        self.load_pkl('', ignore_errors=True)
        self.use_osc = False
        self.osc_addresses = ""

        self.browser = browse_widget.BrowseWidget(viz, "Find", os.path.abspath("."), [".pkl"], width=self.viz.app.button_w, multiple=False, traverse_folders=False)

        for pkl in os.listdir("./models"):
            if pkl.endswith(".pkl"):
                print(pkl, os.path.join(os.getcwd(),"models",pkl))
                self.browse_cache.append(os.path.join(os.getcwd(),"models",pkl))

    def get_params(self):
        return (self.recent_pkls, self.browse_cache, self.cur_pkl, self.user_pkl, self.use_osc, self.osc_addresses)

    def set_params(self, params):
        self.recent_pkls, self.browse_cache, self.cur_pkl, self.user_pkl, self.use_osc, self.osc_addresses = params

    def add_recent(self, pkl, ignore_errors=False):
        try:
            resolved = self.resolve_pkl(pkl)
            if resolved not in self.recent_pkls:
                self.recent_pkls.append(resolved)
        except:
            if not ignore_errors:
                raise

    def load_pkl(self, pkl, ignore_errors=False):
        viz = self.viz
        viz.app.skip_frame() # The input field will change on next frame.
        print(os.getcwd())
        try:
            resolved = self.resolve_pkl(pkl)
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
            if os.path.exists(os.getcwd() + os.sep + 'models' + os.sep + tail):
                self.user_pkl = os.getcwd() + os.sep + 'models' + os.sep + tail
            else:
                print('ERROR: Model does not exist in the model folder.')
        if not os.path.exists(self.cur_pkl):
            head, tail = os.path.split(self.cur_pkl)
            if os.path.exists(os.getcwd() + os.sep + 'models' + os.sep + tail):
                self.cur_pkl = os.getcwd() + os.sep + 'models' + os.sep + tail
            else:
                print('ERROR: Model does not exist in the model folder.')
        for recent_pkl in self.recent_pkls:
            if not os.path.exists(self.recent_pkl):
                head, tail = os.path.split(self.recent_pkl)
                if os.path.exists(os.getcwd() + os.sep + 'models' + os.sep + tail):
                    self.cur_pkl = os.getcwd() + os.sep + 'models' + os.sep + tail
                else:
                    print('ERROR: Model does not exist in the model folder.')
        


    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        recent_pkls = [pkl for pkl in self.recent_pkls if pkl != self.user_pkl]
        if show:
            imgui.text('Pickle')
            imgui.same_line(viz.app.label_w)
            changed, self.user_pkl = imgui_utils.input_text('##pkl', self.user_pkl, 1024,
                flags=(imgui.INPUT_TEXT_AUTO_SELECT_ALL | imgui.INPUT_TEXT_ENTER_RETURNS_TRUE),
                width=(-1 - viz.app.button_w * 3 - viz.app.spacing * 3),
                help_text='<PATH> | <URL> | <RUN_DIR> | <RUN_ID> | <RUN_ID>/<KIMG>.pkl')
            if changed:
                self.load_pkl(self.user_pkl, ignore_errors=True)
            if imgui.is_item_hovered() and not imgui.is_item_active() and self.user_pkl != '':
                imgui.set_tooltip(self.user_pkl)
            imgui.same_line()
            if imgui_utils.button('Recent...', width=viz.app.button_w, enabled=(len(recent_pkls) != 0)):
                imgui.open_popup('recent_pkls_popup')
            imgui.same_line()
            _clicked, pkl = self.browser(self.viz.app.button_w)
            if _clicked:
                print("SELECTED", pkl)
                self.load_pkl(pkl[0], ignore_errors=True)
            imgui.same_line()
            if imgui_utils.button('Models', enabled=len(self.browse_cache) > 0, width=-1):
                imgui.open_popup('browse_pkls_popup')
                self.browse_refocus = True

        if imgui.begin_popup('recent_pkls_popup'):
            for pkl in recent_pkls:
                clicked, _state = imgui.menu_item(pkl)
                if clicked:
                    self.load_pkl(pkl, ignore_errors=True)
            imgui.end_popup()

        if imgui.begin_popup('browse_pkls_popup'):
            for pkl in self.browse_cache:
                clicked, _state = imgui.menu_item(pkl)
                if clicked:
                    self.load_pkl(pkl, ignore_errors=True)

            if self.browse_refocus:
                imgui.set_scroll_here()
                viz.app.skip_frame()  # Focus will change on next frame.
                self.browse_refocus = False

            imgui.end_popup()

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

    def resolve_pkl(self, pattern):
        assert isinstance(pattern, str)
        assert pattern != ''

        # URL => return as is.
        if dnnlib.util.is_url(pattern):
            return pattern

        # Short-hand pattern => locate.
        path = _locate_results(pattern)

        # Run dir => pick the last saved snapshot.
        if os.path.isdir(path):
            pkl_files = sorted(glob.glob(os.path.join(path, 'network-snapshot-*.pkl')))
            if len(pkl_files) == 0:
                raise IOError(f'No network pickle found in "{path}"')
            path = pkl_files[-1]

        # Normalize.
        path = os.path.abspath(path)
        return path

#----------------------------------------------------------------------------
