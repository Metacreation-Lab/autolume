import logging
import queue
import threading

import imgui
import torch

from assets import ACTIVE_RED

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

from features import store
from features.worker import FeatureExtractionJob
from utils.gui_utils import imgui_utils
from widgets.adjuster_state import pack_state, unpack_state
from widgets.osc_menu import osc_address_picker

logger = logging.getLogger(__name__)

DEFAULT_DIM = 512
NUM_FEATURES = 8


class AdjusterWidget:

    def __init__(self, viz):
        self.viz = viz
        self.model_pkl = None
        self.feature_status = 'none'  # none | looking | ready | extracting | error
        self.error_message = ""
        self.job = None
        self._lookup_results = queue.Queue()
        self._lookup_generation = 0
        self.vslide_address = []
        self._reset_sliders(torch.randn(NUM_FEATURES, DEFAULT_DIM))

    # ---------- state ----------

    def _unmap_osc(self):
        for i, address in enumerate(self.vslide_address):
            if address:
                try:
                    self.viz.osc_dispatcher.unmap(f"/{address}", self.vec_handler(i))
                except Exception:
                    logger.warning("OSC address %s is not mapped", address)

    def _reset_sliders(self, dirs, from_features=False):
        self._unmap_osc()
        self.dirs = dirs.to(torch.float32)
        self.base_dirs = self.dirs.clone()
        n = len(self.dirs)
        self.weights = torch.zeros(n)
        self.base_is_feature = [from_features] * n
        self.vslide_use_osc = [False] * n
        self.vslide_address = [""] * n
        self.vslide_mappings = ["x"] * n

    def _add_slider(self):
        new_dir = torch.randn(1, self.dirs.shape[1])
        self.dirs = torch.cat((self.dirs, new_dir), dim=0)
        self.base_dirs = torch.cat((self.base_dirs, new_dir.clone()), dim=0)
        self.weights = torch.cat((self.weights, torch.tensor([0.0])))
        self.base_is_feature.append(False)
        self.vslide_use_osc.append(False)
        self.vslide_address.append("")
        self.vslide_mappings.append("x")

    def _slider_is_feature(self, i):
        return (self.base_is_feature[i]
                and torch.equal(self.dirs[i], self.base_dirs[i]))

    def _current_w_dim(self):
        try:
            return int(self.viz.result.get('w_dim', DEFAULT_DIM))
        except Exception:
            return DEFAULT_DIM

    # ---------- feature lookup and extraction ----------

    def _start_lookup(self, pkl, apply_dirs=True):
        self.feature_status = 'looking'
        self.error_message = ""
        self._lookup_generation += 1
        generation = self._lookup_generation

        def run():
            feature_set = None
            if pkl:
                feature_set = store.lookup(pkl)
            self._lookup_results.put((generation, pkl, feature_set, apply_dirs))

        threading.Thread(target=run, daemon=True, name='feature-lookup').start()

    def _consume_lookup(self):
        try:
            generation, pkl, feature_set, apply_dirs = self._lookup_results.get_nowait()
        except queue.Empty:
            return
        if generation != self._lookup_generation or pkl != self.model_pkl:
            return  # stale result from a superseded lookup
        if feature_set is not None:
            if apply_dirs:
                self._reset_sliders(torch.from_numpy(feature_set.directions),
                                    from_features=True)
            self.feature_status = 'ready'
        else:
            self.feature_status = 'none'
            if apply_dirs:
                self._reset_sliders(torch.randn(NUM_FEATURES, self._current_w_dim()))

    def _poll_job(self):
        if self.job is None:
            return
        self.job.poll()
        if self.job.error is not None:
            self.error_message = self.job.error
            self.feature_status = 'error'
            self.job = None
        elif self.job.done:
            self.job = None
            self._start_lookup(self.model_pkl)

    # ---------- OSC ----------

    def vec_handler(self, idx):
        def func(address, *args):
            try:
                if idx < len(self.weights) and self.vslide_use_osc[idx]:
                    f = lambda x: eval(self.vslide_mappings[idx])
                    out = f(args[-1])
                    if isinstance(out, (int, float)):
                        self.weights[idx] = out
            except Exception as e:
                logger.warning("OSC vector handler failed: %s", e)

        return func

    # ---------- presets ----------

    def save(self, path):
        state = pack_state(self.model_pkl, self.dirs, self.base_dirs, self.weights,
                           self.vslide_use_osc, self.vslide_address,
                           self.vslide_mappings, self.base_is_feature)
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path):
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except Exception:
            logger.warning("Could not read adjuster state from %s", path)
            return
        state = unpack_state(data)
        if state is None:
            logger.warning("Ignoring incompatible adjuster preset at %s", path)
            return
        self._unmap_osc()
        self.model_pkl = state["model_pkl"]
        self.dirs = state["dirs"]
        self.base_dirs = state["base_dirs"]
        self.weights = state["weights"]
        self.base_is_feature = state["base_is_feature"]
        self.vslide_use_osc = state["use_osc"]
        self.vslide_mappings = state["mappings"]
        self.vslide_address = state["addresses"]
        for i, address in enumerate(self.vslide_address):
            if self.vslide_use_osc[i] and address and address != "...":
                self.viz.osc_dispatcher.map(f"/{address}", self.vec_handler(i))
        # Refresh the status label without touching the restored directions.
        self._start_lookup(self.model_pkl, apply_dirs=False)

    # ---------- UI ----------

    def _extract_header(self):
        button_w = max(self.viz.app.button_w,
                       imgui.calc_text_size("Extract features").x
                       + self.viz.app.font_size)
        avail = imgui.get_content_region_available_width()
        if self.job is not None:
            imgui.progress_bar(self.job.progress,
                               (avail - button_w - self.viz.app.spacing, 0),
                               self.job.message)
            imgui.same_line()
            if imgui_utils.button("Cancel##features", width=button_w):
                self.job.cancel()
                self.job = None
                self._start_lookup(self.model_pkl)
        else:
            labels = {
                'ready': "Features loaded",
                'none': "No features extracted for this model",
                'looking': "Checking for features",
                'extracting': "Extracting features",
                'error': self.error_message or "Extraction failed",
            }
            imgui.text(labels[self.feature_status])
            randomize_w = self.viz.app.button_w
            add_w = self.viz.app.button_w
            imgui.same_line(avail - button_w - randomize_w - add_w
                            - 2 * self.viz.app.spacing)
            if imgui_utils.button("Add input##vecs", width=add_w):
                self._add_slider()
            imgui.same_line(avail - button_w - randomize_w - self.viz.app.spacing)
            if imgui_utils.button("Randomize all##vecs", width=randomize_w):
                self.dirs = torch.randn(self.dirs.shape)
            imgui.same_line(avail - button_w)
            button_label = "Re-extract" if self.feature_status == 'ready' \
                else "Extract features"
            enabled = (self.model_pkl is not None
                       and self.feature_status in ('ready', 'none', 'error'))
            if imgui_utils.button(f"{button_label}##features", width=button_w,
                                  enabled=enabled):
                self.job = FeatureExtractionJob(self.model_pkl)
                self.feature_status = 'extracting'
        imgui.separator()

    def _slider_column(self, i):
        s = imgui.get_style()
        draw_list = imgui.get_window_draw_list()
        draw_list.channels_split(2)
        draw_list.channels_set_current(1)
        changed, self.vslide_use_osc[i] = imgui.checkbox(
            f"Use OSC##vslide_use_osc{i}", self.vslide_use_osc[i])
        use_osc = self.vslide_use_osc[i]
        if use_osc:
            draw_list.channels_set_current(0)
            p_min = imgui.get_item_rect_min()
            p_max = imgui.get_item_rect_max()
            draw_list.add_rect_filled(
                p_min.x + (self.viz.app.font_size * 1.5), p_min.y,
                p_min.x + self.viz.app.button_w, p_max.y,
                imgui.get_color_u32_rgba(*ACTIVE_RED))
        draw_list.channels_merge()

        with imgui_utils.grayed_out(not use_osc):
            changed, new_address = imgui_utils.input_text(
                f"##vslide_osc{i}", self.vslide_address[i], 256,
                imgui.INPUT_TEXT_CHARS_NO_BLANK,
                width=self.viz.app.button_w - imgui.get_frame_height()
                - self.viz.app.spacing,
                help_text="osc address")
            imgui.same_line()
            picked, picked_address = osc_address_picker(
                self.viz, f"vslide_osc{i}", self.vslide_address[i],
                enabled=use_osc)
            if picked:
                changed, new_address = True, picked_address
            s.item_spacing = [self.viz.app.spacing, self.viz.app.spacing]
            s.item_inner_spacing = [self.viz.app.spacing, self.viz.app.spacing]
            if changed:
                self.viz.osc_dispatcher.map(f"/{new_address}", self.vec_handler(i))
                try:
                    self.viz.osc_dispatcher.unmap(f"/{self.vslide_address[i]}",
                                                  self.vec_handler(i))
                except Exception:
                    logger.warning("OSC address %s is not mapped",
                                   self.vslide_address[i])
                self.vslide_address[i] = new_address
            changed, self.vslide_mappings[i] = imgui_utils.input_text(
                f"##vslide_mapping{i}", self.vslide_mappings[i], 256,
                flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE,
                width=self.viz.app.button_w, help_text="mapping")

        is_feature = self._slider_is_feature(i)
        label = "feature" if is_feature else "random"
        offset = (self.viz.app.button_w - imgui.calc_text_size(label).x) / 2
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + max(0.0, offset))
        if is_feature:
            imgui.text(label)
        else:
            imgui.text_disabled(label)
        changed, value = imgui.v_slider_float(
            f"##slider{i}", self.viz.app.button_w,
            self.viz.app.content_height / 10, self.weights[i],
            -5, 5, format='%.3f', power=3)
        if changed:
            self.weights[i] = value
        if imgui_utils.button(f"Randomize##{i}", self.viz.app.button_w):
            self.dirs[i] = torch.randn(self.dirs.shape[1])
        half = self.viz.app.button_w * (6 / 8) - (self.viz.app.spacing / 2)
        if imgui_utils.button(f"Reset##{i}", half):
            self.dirs[i] = self.base_dirs[i].clone()
            self.weights[i] = 0
        imgui.same_line()
        if imgui_utils.button(
                f"-##vecs{i}",
                self.viz.app.button_w * (2 / 8) - (self.viz.app.spacing / 2)):
            return i
        return None

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        cur_pkl = self.viz.pickle_widget.cur_pkl
        if cur_pkl != self.model_pkl:
            self.model_pkl = cur_pkl
            if self.job is not None:
                self.job.cancel()
                self.job = None
            self._start_lookup(cur_pkl)
        self._consume_lookup()
        self._poll_job()
        if (self.feature_status == 'none' and self.job is None
                and self.dirs.shape[1] != self._current_w_dim()):
            self._reset_sliders(torch.randn(NUM_FEATURES, self._current_w_dim()))

        if show:
            self._extract_header()
            remove_idx = None
            for i in range(len(self.weights)):
                imgui.begin_group()
                removed = self._slider_column(i)
                if removed is not None:
                    remove_idx = removed
                imgui.end_group()
                if (i < len(self.weights) - 1
                        and (self.viz.pane_w - imgui.get_item_rect_max()[0])
                        >= self.viz.app.button_w):
                    imgui.same_line()

            if remove_idx is not None:
                idxs = [i for i in range(len(self.weights)) if i != remove_idx]
                self.weights = self.weights[idxs]
                self.dirs = self.dirs[idxs]
                self.base_dirs = self.base_dirs[idxs]
                self.base_is_feature = [self.base_is_feature[i] for i in idxs]
                self.vslide_use_osc = [self.vslide_use_osc[i] for i in idxs]
                self.vslide_address = [self.vslide_address[i] for i in idxs]
                self.vslide_mappings = [self.vslide_mappings[i] for i in idxs]

        self.viz.args.update(direction=self.weights @ self.dirs)
