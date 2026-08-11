"""Feature Mixer: a rack of eight patchable direction slots.

Each slot patches one direction from the model's extracted bank, applies it to
a zone of the synthesis layers, and scales it in units of that direction's own
standard deviation.
"""
import logging
import queue
import threading

import imgui
import torch

from assets import ACTIVE_RED, OPAQUEGREEN

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

from features import store
from features.worker import FeatureExtractionJob
from features.zones import (CUSTOM_ZONE, ZONE_LABELS, ZONES, block_labels,
                            layer_mask, match_zone)
from utils.gui_utils import imgui_utils
from widgets.adjuster_state import NUM_SLOTS, make_slot, pack_state, unpack_state
from widgets.osc_menu import osc_address_picker

logger = logging.getLogger(__name__)

DEFAULT_DIM = 512
DEFAULT_NUM_WS = 18
SLIDER_RANGE = 3.0

ZONE_ITEMS = [ZONE_LABELS[zone] for zone in ZONES]


def _random_direction(dim):
    direction = torch.randn(max(1, int(dim)))
    norm = float(direction.norm())
    if norm > 0:
        direction = direction / norm
    return direction


def _random_sigma(dim):
    """Legacy gearing: weight 1 moves w as far as the old unnormalized randn."""
    return float(dim) ** 0.5


def _slot_mask(slot, num_ws):
    if slot["zone"] == CUSTOM_ZONE and slot["layers"]:
        layers = slot["layers"]
        mask = torch.zeros(num_ws, dtype=torch.bool)
        for layer in range(min(num_ws, len(layers))):
            mask[layer] = bool(layers[layer])
        return mask
    try:
        return layer_mask(slot["zone"], num_ws)
    except ValueError:
        return layer_mask("all", num_ws)


class AdjusterWidget:

    def __init__(self, viz):
        self.viz = viz
        self.model_pkl = None
        self.feature_status = 'none'  # none | looking | extracting | ready | error
        self.error_message = ""
        self.job = None
        self.slots_locked = False
        self.bank_dirs = None
        self.bank_sigmas = None
        self.slots = []
        self._auto_attempted = None
        self._handlers = {}
        self._mapped = {}
        self._lookup_results = queue.Queue()
        self._lookup_generation = 0
        self._random_slots(DEFAULT_DIM)

    # ---------- slots ----------

    def _random_slots(self, w_dim):
        self._unmap_all_osc()
        sigma = _random_sigma(w_dim)
        self.slots = [make_slot(_random_direction(w_dim), sigma=sigma)
                      for _ in range(NUM_SLOTS)]

    def _bank_slots(self):
        """Default patch: the first components of the bank, one per slot."""
        self._unmap_all_osc()
        w_dim = int(self.bank_dirs.shape[1])
        slots = []
        for i in range(NUM_SLOTS):
            if i < len(self.bank_dirs):
                slots.append(make_slot(self.bank_dirs[i].clone(), component=i,
                                       sigma=float(self.bank_sigmas[i])))
            else:
                slots.append(make_slot(_random_direction(w_dim),
                                       sigma=_random_sigma(w_dim)))
        self.slots = slots

    def _bank_size(self):
        return 0 if self.bank_dirs is None else int(self.bank_dirs.shape[0])

    def _select_component(self, i, j):
        slot = self.slots[i]
        slot["component"] = j
        slot["direction"] = self.bank_dirs[j].clone()
        slot["sigma"] = float(self.bank_sigmas[j])
        slot["weight"] = 0.0

    def _randomize_slot(self, i):
        slot = self.slots[i]
        dim = slot["direction"].shape[0]
        slot["direction"] = _random_direction(dim)
        slot["component"] = None
        slot["sigma"] = _random_sigma(dim)

    def _reset_slot(self, i):
        slot = self.slots[i]
        component = slot["component"]
        if component is not None and component < self._bank_size():
            slot["direction"] = self.bank_dirs[component].clone()
            slot["sigma"] = float(self.bank_sigmas[component])
        slot["weight"] = 0.0

    def _slot_width(self):
        """Split the pane so all slots fit on a single row when possible."""
        style = imgui.get_style()
        avail = imgui.get_content_region_available_width()
        fit = (avail - (NUM_SLOTS - 1) * style.item_spacing.x) / NUM_SLOTS
        min_w = (imgui.calc_text_size("Randomize").x
                 + 2 * style.frame_padding.x + style.item_spacing.x)
        return min(self.viz.app.button_w, max(fit, min_w))

    def _current_w_dim(self):
        try:
            return int(self.viz.result.get('w_dim', DEFAULT_DIM))
        except Exception:
            return DEFAULT_DIM

    def _current_num_ws(self):
        try:
            return int(self.viz.result.get('num_ws', DEFAULT_NUM_WS))
        except Exception:
            return DEFAULT_NUM_WS

    # ---------- feature lookup and extraction ----------

    def _start_lookup(self, pkl, apply_dirs=True):
        self.feature_status = 'looking'
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
            self.bank_dirs = torch.from_numpy(feature_set.directions).to(torch.float32)
            self.bank_sigmas = torch.from_numpy(feature_set.sigmas).to(torch.float32)
            if apply_dirs:
                self._bank_slots()
            self.feature_status = 'ready'
            self.error_message = ""
            self.slots_locked = False
            return
        self.bank_dirs = None
        self.bank_sigmas = None
        if pkl and self._auto_attempted != pkl:
            if apply_dirs:
                self._random_slots(self._current_w_dim())
            self._start_job(owns_slots=apply_dirs)
            return
        self.slots_locked = False
        if not pkl:
            self.feature_status = 'none'
        elif self._auto_attempted == pkl:
            self.feature_status = 'error'
            self.error_message = self.error_message or "Extraction failed"
        else:
            self.feature_status = 'none'

    def _start_job(self, owns_slots):
        if not self.model_pkl:
            return
        self.job = FeatureExtractionJob(self.model_pkl)
        self.slots_locked = owns_slots
        self.feature_status = 'extracting'
        self.error_message = ""
        self._auto_attempted = self.model_pkl

    def _drop_job(self):
        if self.job is not None:
            self.job.cancel()
            self.job = None
        self.slots_locked = False

    def _poll_job(self):
        if self.job is None:
            return
        self.job.poll()
        if self.job.error is not None:
            self.error_message = self.job.error
            self.feature_status = 'error'
            self.job = None
            self.slots_locked = False
        elif self.job.done:
            self.job = None
            # Ownership persists through the follow up lookup that applies it.
            self._start_lookup(self.model_pkl, apply_dirs=self.slots_locked)

    # ---------- OSC ----------

    def vec_handler(self, idx):
        """Stable per slot handler so map and unmap stay symmetric."""
        handler = self._handlers.get(idx)
        if handler is not None:
            return handler

        def func(address, *args):
            try:
                if idx < len(self.slots) and self.slots[idx]["use_osc"]:
                    f = lambda x: eval(self.slots[idx]["mapping"])
                    out = f(args[-1])
                    if isinstance(out, (int, float)):
                        self.slots[idx]["weight"] = float(out)
            except Exception as e:
                logger.warning("OSC vector handler failed: %s", e)

        self._handlers[idx] = func
        return func

    def _map_osc(self, i, address):
        self._unmap_osc(i)
        if not address or address == "...":
            return
        self.viz.osc_dispatcher.map(f"/{address}", self.vec_handler(i))
        self._mapped[i] = address

    def _unmap_osc(self, i):
        address = self._mapped.pop(i, None)
        if not address:
            return
        try:
            self.viz.osc_dispatcher.unmap(f"/{address}", self.vec_handler(i))
        except Exception:
            logger.warning("OSC address %s is not mapped", address)

    def _unmap_all_osc(self):
        for i in list(self._mapped):
            self._unmap_osc(i)

    # ---------- presets ----------

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(pack_state(self.model_pkl, self.slots), f)

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
        self._unmap_all_osc()
        self.model_pkl = state["model_pkl"]
        self.slots = state["slots"]
        self.bank_dirs = None
        self.bank_sigmas = None
        self.error_message = ""
        # A running job keeps running but must not overwrite the preset slots.
        self.slots_locked = False
        for i, slot in enumerate(self.slots):
            if slot["use_osc"] and slot["address"] and slot["address"] != "...":
                self._map_osc(i, slot["address"])
        # Refresh the bank without touching the restored slots.
        self._start_lookup(self.model_pkl, apply_dirs=False)

    # ---------- UI ----------

    def _header(self):
        """Only shown while extracting or after a failure. Silent otherwise."""
        if self.job is None and self.feature_status != 'error':
            return
        viz = self.viz
        button_w = viz.app.button_w
        style = imgui.get_style()
        spacing = style.item_spacing.x
        avail = imgui.get_content_region_available_width()
        button_x = avail - button_w
        if self.job is not None:
            bar_w = max(button_w, button_x - 2 * spacing)
            imgui.progress_bar(self.job.progress, (bar_w, 0), self.job.message)
            imgui.same_line(button_x)
            if imgui_utils.button("Cancel##features", width=button_w):
                self._drop_job()
                self.feature_status = 'error'
                self.error_message = "Extraction cancelled"
        else:
            imgui.text(self.error_message or "Extraction failed")
            if self.model_pkl:
                imgui.same_line(button_x)
                if imgui_utils.button("Retry##features", width=button_w):
                    self._start_job(owns_slots=True)
        imgui.separator()

    def _seed_layers(self, i):
        """Fill the slot's layer list from its zone so the grid shows it."""
        slot = self.slots[i]
        num_ws = max(1, self._current_num_ws())
        if slot["zone"] == CUSTOM_ZONE:
            layers = list(slot["layers"] or [])
            layers = ([bool(v) for v in layers] + [False] * num_ws)[:num_ws]
        else:
            zone = slot["zone"] if slot["zone"] in ZONES else "all"
            layers = layer_mask(zone, num_ws).tolist()
        slot["layers"] = layers

    def _sync_zone(self, i):
        """The zone always names the selected layers: a preset or custom."""
        slot = self.slots[i]
        slot["zone"] = match_zone(slot["layers"], max(1, self._current_num_ws()))

    def _zone_popup(self, i, enabled):
        slot = self.slots[i]
        mouse = imgui.get_mouse_position()
        imgui.set_next_window_position(mouse.x, mouse.y,
                                       condition=imgui.APPEARING)
        imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND, *OPAQUEGREEN)
        imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND_ACTIVE, *OPAQUEGREEN)
        imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND_COLLAPSED, *OPAQUEGREEN)
        opened, visible = imgui_utils.begin_popup_modal(
            f"Layers##zone{i}", visible=True,
            flags=imgui.WINDOW_NO_RESIZE | imgui.WINDOW_ALWAYS_AUTO_RESIZE)
        imgui.pop_style_color(3)
        if not opened:
            return
        num_ws = max(1, self._current_num_ws())
        labels = block_labels(num_ws)
        layers = ([bool(v) for v in (slot["layers"] or [])]
                  + [False] * num_ws)[:num_ws]
        spacing = self.viz.app.spacing
        frame = imgui.get_frame_height()
        label_col = (frame + 2 * spacing
                     + max(imgui.calc_text_size(label).x for label in labels)
                     + spacing)
        content_w = label_col + 2 * frame + spacing
        for b, label in enumerate(labels):
            first = 2 * b
            if first >= num_ws:
                break
            row = [layer for layer in (first, first + 1) if layer < num_ws]
            changed, value = imgui.checkbox(f"##zone_row{i}_{b}",
                                            all(layers[layer] for layer in row))
            if changed and enabled:
                for layer in row:
                    layers[layer] = value
                slot["layers"] = list(layers)
                self._sync_zone(i)
            imgui.same_line()
            imgui.text(label)
            imgui.same_line(label_col)
            for layer in row:
                if layer != first:
                    imgui.same_line()
                changed, value = imgui.checkbox(f"##layer{i}_{layer}",
                                                layers[layer])
                if changed and enabled:
                    layers[layer] = value
                    slot["layers"] = list(layers)
                    self._sync_zone(i)
        imgui.separator()
        half = (content_w - spacing) / 2
        if imgui_utils.button(f"None##zone{i}", width=half, enabled=enabled):
            slot["layers"] = [False] * num_ws
            self._sync_zone(i)
        imgui.same_line()
        if imgui_utils.button(f"All##zone{i}", width=half, enabled=enabled):
            slot["layers"] = [True] * num_ws
            self._sync_zone(i)
        if not visible:
            imgui.close_current_popup()
        imgui.end_popup()

    def _slot_osc(self, i, enabled, slot_w):
        viz = self.viz
        slot = self.slots[i]
        s = imgui.get_style()
        draw_list = imgui.get_window_draw_list()
        draw_list.channels_split(2)
        draw_list.channels_set_current(1)
        changed, use_osc = imgui.checkbox(f"Use OSC##slot_use_osc{i}",
                                          slot["use_osc"])
        if changed and enabled:
            slot["use_osc"] = use_osc
        use_osc = slot["use_osc"]
        if use_osc:
            draw_list.channels_set_current(0)
            p_min = imgui.get_item_rect_min()
            p_max = imgui.get_item_rect_max()
            draw_list.add_rect_filled(
                p_min.x + (viz.app.font_size * 1.5), p_min.y,
                p_min.x + slot_w, p_max.y,
                imgui.get_color_u32_rgba(*ACTIVE_RED))
        draw_list.channels_merge()

        with imgui_utils.grayed_out(not use_osc):
            changed, new_address = imgui_utils.input_text(
                f"##slot_osc{i}", slot["address"], 256,
                imgui.INPUT_TEXT_CHARS_NO_BLANK,
                width=slot_w - imgui.get_frame_height()
                - viz.app.spacing,
                help_text="osc address")
            imgui.same_line()
            picked, picked_address = osc_address_picker(
                viz, f"slot_osc{i}", slot["address"],
                enabled=use_osc and enabled)
            if picked:
                changed, new_address = True, picked_address
            s.item_spacing = [viz.app.spacing, viz.app.spacing]
            s.item_inner_spacing = [viz.app.spacing, viz.app.spacing]
            if changed and enabled:
                self._map_osc(i, new_address)
                slot["address"] = new_address
            changed, mapping = imgui_utils.input_text(
                f"##slot_mapping{i}", slot["mapping"], 256,
                flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE,
                width=slot_w, help_text="mapping")
            if changed and enabled:
                slot["mapping"] = mapping

    def _slot_column(self, i, enabled, slot_w):
        viz = self.viz
        slot = self.slots[i]
        button_w = slot_w

        with imgui_utils.grayed_out(not enabled):
            items = ["random"] + [f"f{j + 1}" for j in range(self._bank_size())]
            current = 0 if slot["component"] is None else slot["component"] + 1
            with imgui_utils.item_width(button_w):
                changed, new_index = imgui.combo(f"##component{i}", current, items)
            if changed and enabled:
                if new_index == 0:
                    if slot["component"] is not None:
                        self._randomize_slot(i)
                else:
                    self._select_component(i, new_index - 1)

            # Custom is shown when the layers match no preset, never offered.
            if slot["zone"] in ZONES:
                items, zone_index = ZONE_ITEMS, ZONES.index(slot["zone"])
            else:
                items = ZONE_ITEMS + [ZONE_LABELS[CUSTOM_ZONE]]
                zone_index = len(ZONES)
            with imgui_utils.item_width(button_w):
                changed, new_index = imgui.combo(f"##zone{i}", zone_index, items)
            if changed and enabled and new_index < len(ZONES):
                slot["zone"] = ZONES[new_index]
                slot["layers"] = None

            changed, value = imgui.v_slider_float(
                f"##slot{i}", button_w, viz.app.content_height / 10,
                slot["weight"], -SLIDER_RANGE, SLIDER_RANGE, format='%.2f')
            if changed and enabled:
                slot["weight"] = float(value)

            if imgui_utils.button(f"Customize##slot{i}", width=button_w,
                                  enabled=enabled):
                self._seed_layers(i)
                imgui.open_popup(f"Layers##zone{i}")
            self._zone_popup(i, enabled)
            if imgui_utils.button(f"Randomize##slot{i}", width=button_w,
                                  enabled=enabled):
                self._randomize_slot(i)
            if imgui_utils.button(f"Reset##slot{i}", width=button_w,
                                  enabled=enabled):
                self._reset_slot(i)

            self._slot_osc(i, enabled, slot_w)

    def _compose(self):
        num_ws = max(1, self._current_num_ws())
        w_dim = max(1, self._current_w_dim())
        offset = torch.zeros(num_ws, w_dim)
        if all(slot["direction"].shape[0] == w_dim for slot in self.slots):
            for slot in self.slots:
                if slot["weight"] != 0.0:
                    offset[_slot_mask(slot, num_ws)] += \
                        slot["weight"] * slot["sigma"] * slot["direction"]
        self.viz.args.update(direction=offset)

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        cur_pkl = self.viz.pickle_widget.cur_pkl
        if cur_pkl != self.model_pkl:
            self.model_pkl = cur_pkl
            self._drop_job()
            self.bank_dirs = None
            self.bank_sigmas = None
            self.error_message = ""
            self._auto_attempted = None
            self._start_lookup(cur_pkl)
        self._consume_lookup()
        self._poll_job()
        if (self.bank_dirs is None and self.job is None
                and self.slots[0]["direction"].shape[0] != self._current_w_dim()):
            self._random_slots(self._current_w_dim())

        if show:
            self._header()
            enabled = not self.slots_locked
            slot_w = self._slot_width()
            for i in range(NUM_SLOTS):
                imgui.begin_group()
                self._slot_column(i, enabled, slot_w)
                imgui.end_group()
                if (i < NUM_SLOTS - 1
                        and (self.viz.pane_w - imgui.get_item_rect_max()[0])
                        >= slot_w):
                    imgui.same_line()

        self._compose()
