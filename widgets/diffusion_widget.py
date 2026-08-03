import logging
import multiprocessing as mp
import os
import re
import time

import imgui

import dnnlib
from diffusion import engine as diffusion_engine
from diffusion import trt
from utils.app_logging import LoggedProcess
from utils.gui_utils import imgui_utils
from utils import session_state
from utils.model_dir import (diffusion_checkpoints_dir, diffusion_loras_dir,
                             list_diffusion_checkpoints, list_diffusion_loras)
from widgets import osc_menu
from widgets.model_dropdown_widget import ModelDropdownButton
from widgets.native_browser_widget import NativeBrowserWidget

logger = logging.getLogger(__name__)

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

MODELS = ["stabilityai/sd-turbo", "KBlueLeaf/kohaku-v2.1"]
ACCELERATIONS = ["none", "tensorrt"]
RESOLUTIONS = [512, 768, 1024]
SESSION_SECTION = "diffusion"

TEXT_LABELS = ('Prompt', 'Strength', 'Checkpoint', 'Resolution', 'Weight')
CHECKBOX_LABELS = ('Enable', 'LoRA', 'TensorRT')

RED = (1.0, 0.3, 0.3, 1.0)
AMBER = (1.0, 0.75, 0.3, 1.0)
GREEN = (0.4, 0.9, 0.5, 1.0)
GRAY = (0.6, 0.6, 0.6, 1.0)


def _short_name(path_or_id):
    if path_or_id.lower().endswith(('.safetensors', '.ckpt')):
        return os.path.splitext(os.path.basename(path_or_id))[0]
    return path_or_id


def build_fraction(message):
    """Rough completion from a build stage message: 3 engines, 2 phases each."""
    match = re.search(r'\((\d+) of (\d+)\)', message)
    if not match:
        return 1.0 if message.startswith('Finishing') else 0.0
    index, total = int(match.group(1)), int(match.group(2))
    return min(1.0, (index - 1 + (0.5 if message.startswith('Compiling') else 0.0)) / total)


def engine_label(entry):
    lora = (f'{_short_name(entry["lora_path"])} @ {entry["lora_scale"]:g}'
            if entry["lora_path"] else 'no LoRA')
    return f'{_short_name(entry["model"])} | {lora} | {entry["resolution"]}'

#----------------------------------------------------------------------------

class DiffusionWidget:
    def __init__(self, viz):
        self.viz = viz
        self.available = diffusion_engine.is_available()
        self.enabled = False
        self.params = dnnlib.EasyDict(diffusion_engine.default_params())
        # kept outside params so unticking LoRA does not lose the path or weight
        self.use_lora = False
        self.lora_path = self.params.lora_path
        self.lora_scale = self.params.lora_scale
        self.prompt_history = session_state.get_recent(SESSION_SECTION, 'prompts')
        if self.prompt_history:
            self.params.prompt = self.prompt_history[0]
        self._prompt_committed = self.params.prompt
        self._prompt_editing = False
        self._lora_scale_ui = self.lora_scale
        self._lora_scale_dragging = False
        self.browser = NativeBrowserWidget()
        os.makedirs(diffusion_checkpoints_dir(), exist_ok=True)
        os.makedirs(diffusion_loras_dir(), exist_ok=True)
        self.model_dropdown = ModelDropdownButton(
            label='Models##diffusion',
            items_provider=lambda: MODELS + list_diffusion_checkpoints(),
            include_training_runs=False)
        self.lora_dropdown = ModelDropdownButton(
            label='LoRAs##diffusion', items_provider=list_diffusion_loras,
            include_training_runs=False)
        self.build_state = 'idle'  # 'idle' | 'building' | 'error'
        self.build_message = ''
        self.build_fraction = 0.0
        self.build_error = ''
        self.build_process = None
        self.build_queue = None
        self.build_reply = None
        self._ready_key = None
        self._ready_checked = 0.0
        self._ready = False
        self._built_engines = []

        funcs = dict(zip(["Prompt", "Strength", "Seed"],
                         [self.osc_handler(param) for param in ["prompt", "strength", "seed"]]))

        self.osc_menu = osc_menu.OscMenu(self.viz, funcs,
                                         label="##DiffusionOSC")

    def osc_handler(self, param):
        def func(address, *args):
            try:
                nec_type = type(self.params[param])
                self.params[param] = nec_type(args[-1])
            except Exception as e:
                logger.warning("OSC handler failed: %s", e)
        return func

    def get_params(self):
        return (self.enabled, dict(self.params), self.use_lora, self.lora_path,
                self.lora_scale, self.osc_menu.get_params())

    def set_params(self, params):
        use_lora = lora_path = lora_scale = None
        if len(params) == 6:
            enabled, saved, use_lora, lora_path, lora_scale, osc_params = params
        elif len(params) == 5:  # legacy: custom-model field and combo index
            enabled, saved, _custom_model, _model_index, osc_params = params
        else:
            enabled, saved, osc_params = params
        self.params = dnnlib.EasyDict(diffusion_engine.default_params())
        self.params.update(saved)
        self.params.strength = float(self.params.strength)
        self.params.seed = int(self.params.seed)
        # older presets encoded "LoRA off" as an empty path
        self.lora_path = self.params.lora_path if lora_path is None else lora_path
        self.lora_scale = float(self.params.lora_scale if lora_scale is None else lora_scale)
        self.use_lora = bool(self.lora_path) if use_lora is None else bool(use_lora)
        self._lora_scale_ui = self.lora_scale
        self.enabled = enabled and self.available
        self.osc_menu.set_params(osc_params)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.get_params(), f)

    def load(self, path):
        with open(path, "rb") as f:
            self.set_params(pickle.load(f))

    def engines_ready(self):
        # the probe hits the disk, so it is only repeated once a second
        key = trt.engine_dir_key(self.params)
        now = time.time()
        if key != self._ready_key or now - self._ready_checked > 1.0:
            self._ready_key = key
            self._ready_checked = now
            self._ready = trt.engines_ready(self.params)
        return self._ready

    def start_build(self):
        self.build_message = 'Starting'
        self.build_error = ''
        self.build_state = 'building'
        self.build_queue = mp.Queue()
        self.build_reply = mp.Queue()
        self.build_process = LoggedProcess(target=trt.run_build, args=(self.build_queue, self.build_reply),
                                           daemon=True, name='trt-build')
        self.build_process.start()
        self.build_queue.put({'cmd': 'build', 'params': dict(self.params)})

    def stop_build(self):
        if self.build_process is not None and self.build_process.is_alive():
            self.build_process.terminate()
        self.build_process = None
        self.build_queue = None
        self.build_reply = None
        self._ready_key = None

    def poll_build(self):
        while self.build_reply is not None and not self.build_reply.empty():
            message = self.build_reply.get()
            if 'error' in message:
                lines = str(message['error']).strip().splitlines()
                self.build_error = lines[-1] if lines else 'Unknown error'
                self.build_state = 'error'
                self.stop_build()
            elif message.get('done'):
                self.build_state = 'idle'
                self.stop_build()
            elif 'progress' in message:
                self.build_message = message['progress']
                self.build_fraction = build_fraction(message['progress'])

    def draw_build_modal(self):
        app = self.viz.app
        width = app.content_width // 2.5
        if self.build_state in ('building', 'error'):
            imgui.open_popup('diffusion_build_modal')
            imgui.set_next_window_position(app.content_width / 2 - width / 2, app.content_height / 3)
            imgui.set_next_window_size(width, 0)
        if imgui.begin_popup_modal('diffusion_build_modal',
                                   flags=imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_MOVE)[0]:
            if self.build_state == 'building':
                imgui.text('First build takes 20 to 30 minutes. The app stays usable.')
                imgui.separator()
                imgui.spacing()
                imgui.text(self.build_message)
                imgui.progress_bar(self.build_fraction, (width - 40, 20))
                imgui.spacing()
                if imgui_utils.button('Cancel##diffusion_build', width=app.button_w):
                    self.stop_build()
                    self.build_state = 'idle'
                    imgui.close_current_popup()
            elif self.build_state == 'error':
                imgui.text('Engine build failed.')
                imgui.text_colored(self.build_error, 1.0, 0.3, 0.3, 1.0)
                imgui.spacing()
                if imgui_utils.button('Close##diffusion_build', width=app.button_w):
                    self.build_state = 'idle'
                    imgui.close_current_popup()
            else:
                imgui.close_current_popup()
            imgui.end_popup()

    def status_line(self, status):
        """(text, rgba). One line, always present, so the panel is never silent.

        Loading and ready only: what is wrong with a specific part of the setup
        belongs on that part's indicator."""
        if not self.available:
            return 'Requires an NVIDIA GPU. Not available on this platform.', GRAY
        if status.startswith('Error'):
            return status, RED
        if status.startswith(diffusion_engine.LOADING_PREFIX):
            return status, AMBER
        if not self.enabled:
            return 'Off - configure below, then enable', GRAY
        return 'Ready', GREEN

    def status_indicators(self, status, loaded, loading):
        """(label, rgba, tooltip) for checkpoint, LoRA and TensorRT.

        Selected is not running: only `loaded`, the params of the pipeline
        actually in VRAM, can say what is really in effect.
        """
        live = loaded or {}
        errored = status.startswith('Error')
        off = not (self.available and self.enabled)

        if off:
            checkpoint = (GRAY, 'Not running')
        elif errored:
            checkpoint = (RED, status)
        elif live.get('model') == self.params.model:
            checkpoint = (GREEN, f'{_short_name(self.params.model)} loaded')
        else:
            checkpoint = (AMBER, f'Loading {_short_name(self.params.model)}'
                          if loading else 'Not loaded')

        if not self.use_lora:
            lora = (GRAY, 'Off - the checkpoint runs unmodified')
        elif not self.lora_path:
            lora = (AMBER, 'No LoRA file selected')
        elif off:
            lora = (GRAY, f'{_short_name(self.lora_path)} - not running')
        elif errored:
            lora = (RED, status)
        elif (live.get('lora_path') == self.lora_path
              and float(live.get('lora_scale', 0.0)) == float(self.lora_scale)):
            lora = (GREEN, f'{_short_name(self.lora_path)} fused at {self.lora_scale:g}')
        else:
            lora = (AMBER, 'Fusing' if loading else 'Not fused')

        # the stage runs unaccelerated when engines are missing, so the label
        # carries the reason rather than stealing the status line for it
        trt_label = 'TensorRT'
        if self.params.acceleration != 'tensorrt':
            tensorrt = (GRAY, 'Off - running the unaccelerated pipeline')
        elif self.build_state == 'building':
            trt_label = 'TensorRT (building)'
            tensorrt = (AMBER, self.build_message or 'Building engines')
        elif not self.engines_ready():
            trt_label = 'TensorRT (unbuilt)'
            tensorrt = (RED, 'No engines for this setup, so it runs unaccelerated. '
                             'Use Build engines.')
        elif live.get('acceleration') == 'tensorrt':
            tensorrt = (GREEN, 'Engines built and running')
        else:
            tensorrt = (AMBER, 'Engines built, not loaded yet')

        return [('Checkpoint',) + checkpoint, ('LoRA',) + lora, (trt_label,) + tensorrt]

    def status_dot_label(self, label, color):
        """A bullet and its label. The bundled font carries no dot glyph, so the
        dot is drawn, and it is centred on the label's own rect: a framed widget
        earlier on the line shifts text down and the dot would not follow."""
        size = imgui.get_text_line_height()
        imgui.begin_group()
        imgui.dummy(size, size)
        dot_x = imgui.get_item_rect_min()[0] + size * 0.5
        imgui.same_line()
        imgui.text_colored(label, *color)
        top, bottom = imgui.get_item_rect_min()[1], imgui.get_item_rect_max()[1]
        imgui.end_group()
        imgui.get_window_draw_list().add_circle_filled(
            dot_x, (top + bottom) * 0.5, size * 0.25, imgui.get_color_u32_rgba(*color))

    def draw_status(self, viz, status, loaded, loading):
        for i, (label, color, tooltip) in enumerate(self.status_indicators(status, loaded, loading)):
            if i:
                imgui.same_line(spacing=viz.app.spacing * 3)
            self.status_dot_label(label, color)
            if imgui.is_item_hovered():
                imgui.set_tooltip(tooltip)

    def column_x(self, viz):
        """X of the control column: past the widest label the panel can draw, so
        rows line up whatever their label says."""
        text_w = max(imgui.calc_text_size(t).x for t in TEXT_LABELS)
        box_w = imgui.get_frame_height() + imgui.get_style().item_spacing[0]
        check_w = box_w + max(imgui.calc_text_size(t).x for t in CHECKBOX_LABELS)
        return max(viz.app.label_w, text_w, check_w) + viz.app.spacing * 2

    def control_column(self, viz):
        """Move to the shared control column, never overlapping the label."""
        imgui.same_line()
        target = self.column_x(viz)
        if imgui.get_cursor_pos_x() < target:
            imgui.set_cursor_pos_x(target)

    def row_label(self, viz, text):
        imgui.text(text)
        self.control_column(viz)

    def commit_prompt_history(self):
        """Record the prompt once the user is done typing, never per keystroke."""
        if self.params.prompt == self._prompt_committed:
            return
        self._prompt_committed = self.params.prompt
        self.prompt_history = session_state.push_recent(
            SESSION_SECTION, 'prompts', self.params.prompt)

    def draw_live_controls(self, viz):
        self.row_label(viz, 'Prompt')
        _changed, self.params.prompt = imgui_utils.input_text(
            '##diffusion_prompt', self.params.prompt, 1024, 0,
            width=-1 - viz.app.button_w - viz.app.spacing,
            help_text="what the image should become")
        if imgui.is_item_active():
            self._prompt_editing = True
        elif self._prompt_editing:
            self._prompt_editing = False
            self.commit_prompt_history()
        imgui.same_line()
        if imgui_utils.button('Recent##diffusion_prompt', width=viz.app.button_w,
                              enabled=bool(self.prompt_history)):
            imgui.open_popup('diffusion_prompt_history')
        if imgui.begin_popup('diffusion_prompt_history'):
            for i, prompt in enumerate(self.prompt_history):
                clicked, _state = imgui.menu_item(f'{prompt[:80]}##prompt{i}')
                if clicked:
                    self.params.prompt = prompt
                    self.commit_prompt_history()
            imgui.end_popup()

        self.row_label(viz, 'Strength')
        with imgui_utils.item_width(-1 - viz.app.button_w * 3 - viz.app.spacing * 3):
            _changed, self.params.strength = imgui.slider_float(
                '##diffusion_strength', self.params.strength, 0, 1, format='%.2f')
        imgui.same_line()
        imgui.text('Seed')
        imgui.same_line()
        with imgui_utils.item_width(-1):
            _changed, self.params.seed = imgui.input_int('##diffusion_seed', self.params.seed)

    def draw_model_setup(self, viz):
        self.row_label(viz, 'Checkpoint')
        changed, model_text = imgui_utils.input_text(
            '##diffusion_model', self.params.model, 1024,
            imgui.INPUT_TEXT_AUTO_SELECT_ALL | imgui.INPUT_TEXT_ENTER_RETURNS_TRUE,
            width=-1 - viz.app.button_w * 2 - viz.app.spacing * 2,
            help_text="model id or checkpoint file")
        if changed:
            self.params.model = model_text.strip()
        if imgui.is_item_hovered() and not imgui.is_item_active() and self.params.model:
            imgui.set_tooltip(self.params.model)
        imgui.same_line()
        if imgui_utils.button('Browse##diffusion_model', width=viz.app.button_w):
            model = self.browser.select_checkpoint_file(
                initial_dir=self.params.model
                if os.path.isfile(self.params.model) else diffusion_checkpoints_dir())
            if model:
                self.params.model = str(model)
        imgui.same_line()
        picked = self.model_dropdown(width=-1)
        if picked is not None:
            self.params.model = picked

        self.row_label(viz, 'Resolution')
        res_index = (RESOLUTIONS.index(self.params.resolution)
                     if self.params.resolution in RESOLUTIONS else 0)
        with imgui_utils.item_width(viz.app.button_w * 1.5):
            changed, res_index = imgui.combo('##diffusion_resolution', res_index,
                                             [str(r) for r in RESOLUTIONS])
        if changed:
            self.params.resolution = RESOLUTIONS[res_index]

        _clicked, self.use_lora = imgui.checkbox('LoRA##diffusion_use', self.use_lora)
        self.control_column(viz)
        with imgui_utils.grayed_out(not self.use_lora):
            changed, lora_text = imgui_utils.input_text(
                '##diffusion_lora', self.lora_path, 1024,
                imgui.INPUT_TEXT_AUTO_SELECT_ALL | imgui.INPUT_TEXT_ENTER_RETURNS_TRUE,
                width=-1 - viz.app.button_w * 2 - viz.app.spacing * 2, help_text="lora file")
            if changed:
                self.lora_path = lora_text.strip()
            if imgui.is_item_hovered() and not imgui.is_item_active() and self.lora_path:
                imgui.set_tooltip(self.lora_path)
            imgui.same_line()
            if imgui_utils.button('Browse##lora', width=viz.app.button_w):
                lora = self.browser.select_lora_file(
                    initial_dir=self.lora_path or diffusion_loras_dir())
                if lora:
                    self.lora_path = str(lora)
                    self.use_lora = True
            imgui.same_line()
            picked = self.lora_dropdown(width=-1)
            if picked is not None:
                self.lora_path = picked
                self.use_lora = True

            self.row_label(viz, 'Weight')
            # slider LoRAs (age, LECO) use weights well past 1, in both directions.
            # a scale change reloads the pipeline, so it only commits on release
            with imgui_utils.item_width(-1):
                changed, scale_ui = imgui.slider_float('##diffusion_lora_scale',
                                                       self._lora_scale_ui, -5, 5, format='%.2f')
            if changed:
                self._lora_scale_ui = scale_ui
            if imgui.is_item_active():
                self._lora_scale_dragging = True
            elif self._lora_scale_dragging:
                self._lora_scale_dragging = False
                self.lora_scale = self._lora_scale_ui
            else:
                self._lora_scale_ui = self.lora_scale

    def draw_acceleration(self, viz, needs_build):
        _clicked, use_trt = imgui.checkbox('TensorRT##diffusion', self.params.acceleration == 'tensorrt')
        self.params.acceleration = 'tensorrt' if use_trt else 'none'
        self.control_column(viz)
        if imgui_utils.button('Build engines##diffusion', width=viz.app.button_w * 1.5,
                              enabled=(needs_build and self.build_state == 'idle')):
            self.start_build()
        imgui.same_line()
        if imgui_utils.button('Saved setups##diffusion', width=viz.app.button_w * 1.5):
            self._built_engines = trt.list_built_engines()
            imgui.open_popup('diffusion_engines_popup')
        if imgui.begin_popup('diffusion_engines_popup'):
            if not self._built_engines:
                imgui.menu_item('No engines built yet', None, False, False)
            for i, entry in enumerate(self._built_engines):
                clicked, _state = imgui.menu_item(f'{engine_label(entry)}##engine{i}')
                if clicked:
                    self.params.model = entry['model']
                    self.lora_path = entry['lora_path']
                    self.use_lora = bool(entry['lora_path'])
                    self.lora_scale = float(entry['lora_scale'])
                    self._lora_scale_ui = self.lora_scale
                    self.params.resolution = int(entry['resolution'])
                    self.params.acceleration = 'tensorrt'
            imgui.end_popup()

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        status = viz.result.get('diffusion_status', '')
        loaded = viz.result.get('diffusion_loaded')
        loading = bool(viz.result.get('diffusion_loading'))
        self.poll_build()
        # the LoRA checkbox keeps the path while off, so params carry the effective value
        self.params.lora_path = self.lora_path if self.use_lora else ''
        self.params.lora_scale = self.lora_scale
        needs_build = (self.params.acceleration == 'tensorrt' and self.available
                       and not self.engines_ready())

        if show:
            with imgui_utils.grayed_out(not self.available):
                _clicked, enabled = imgui.checkbox('Enable##diffusion', self.enabled)
                if self.available:
                    self.enabled = enabled
                self.control_column(viz)
                self.draw_status(viz, status, loaded, loading)
                # end_group() already broke the line, so no same_line() here
                imgui.set_cursor_pos_x(self.column_x(viz))
                text, color = self.status_line(status)
                imgui.text_colored(text, *color)

                # everything stays editable while disabled: set the look up first,
                # then Enable starts processing with it. Nothing waits on a nudge.
                self.draw_live_controls(viz)
                imgui.spacing()
                imgui.separator()
                imgui.text_colored('Model setup', *GRAY)
                self.draw_model_setup(viz)
                imgui.spacing()
                self.draw_acceleration(viz, needs_build)

            self.osc_menu()

        self.draw_build_modal()
        self.params.strength = float(min(max(self.params.strength, 0.0), 1.0))
        self.params.seed = int(self.params.seed)
        self.params.lora_scale = float(min(max(self.params.lora_scale, -5.0), 5.0))
        if self.params.acceleration not in ACCELERATIONS:
            self.params.acceleration = ACCELERATIONS[0]
        viz.args.use_diffusion = bool(self.enabled and self.available)
        viz.args.diffusion = dict(self.params)

#----------------------------------------------------------------------------
