import logging
import math
import multiprocessing as mp
import time

import imgui

import dnnlib
from diffusion import engine as diffusion_engine
from diffusion import trt
from utils.app_logging import LoggedProcess
from utils.gui_utils import imgui_utils
from widgets import osc_menu
from widgets.native_browser_widget import NativeBrowserWidget

logger = logging.getLogger(__name__)

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

MODELS = ["stabilityai/sd-turbo", "KBlueLeaf/kohaku-v2.1"]
ACCELERATIONS = ["none", "tensorrt"]
ACCELERATION_LABELS = ["Standard", "TensorRT"]

#----------------------------------------------------------------------------

class DiffusionWidget:
    def __init__(self, viz):
        self.viz = viz
        self.available = diffusion_engine.is_available()
        self.enabled = False
        self.params = dnnlib.EasyDict(diffusion_engine.default_params())
        self.custom_model = ""
        self.model_index = 0
        self.browser = NativeBrowserWidget()
        self.build_state = 'idle'  # 'idle' | 'building' | 'error'
        self.build_message = ''
        self.build_error = ''
        self.build_process = None
        self.build_queue = None
        self.build_reply = None
        self._ready_key = None
        self._ready_checked = 0.0
        self._ready = False

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
        return self.enabled, dict(self.params), self.custom_model, self.model_index, self.osc_menu.get_params()

    def set_params(self, params):
        enabled, saved, self.custom_model, self.model_index, osc_params = params
        self.params = dnnlib.EasyDict(diffusion_engine.default_params())
        self.params.update(saved)
        self.params.strength = float(self.params.strength)
        self.params.seed = int(self.params.seed)
        self.model_index = min(max(int(self.model_index), 0), len(MODELS))
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
                imgui.progress_bar(math.fmod(time.time(), 1.0), (width - 40, 20))
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

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        status = viz.result.get('diffusion_status', '')
        self.poll_build()

        if show:
            with imgui_utils.grayed_out(not self.available):
                _clicked, enabled = imgui.checkbox('Enable##diffusion', self.enabled)
                if self.available:
                    self.enabled = enabled
                else:
                    imgui.same_line()
                    imgui.text('Requires an NVIDIA GPU. Not available on this platform.')

                with imgui_utils.grayed_out(not self.enabled):
                    with imgui_utils.item_width(viz.app.button_w * 3):
                        _changed, self.model_index = imgui.combo('##diffusion_model', self.model_index,
                                                                 MODELS + ['Custom'])
                    if self.model_index == len(MODELS):
                        imgui.same_line()
                        _changed, self.custom_model = imgui_utils.input_text('##diffusion_custom_model',
                                                                            self.custom_model, 256,
                                                                            imgui.INPUT_TEXT_CHARS_NO_BLANK,
                                                                            width=-1, help_text="model id")
                        self.params.model = self.custom_model.strip() or self.params.model
                    else:
                        self.params.model = MODELS[self.model_index]

                    _changed, self.params.prompt = imgui_utils.input_text(
                        '##diffusion_prompt', self.params.prompt, 1024, 0,
                        width=-1 - viz.app.font_size * 4 - viz.app.spacing, help_text="prompt")
                    imgui.same_line()
                    imgui.text('Prompt')

                    with imgui_utils.item_width(viz.app.button_w * 2):
                        _changed, self.params.strength = imgui.slider_float('##diffusion_strength',
                                                                            self.params.strength, 0, 1,
                                                                            format='Strength %.2f')
                    imgui.same_line(spacing=viz.app.spacing * 2)
                    with imgui_utils.item_width(viz.app.button_w * 1.5):
                        _changed, self.params.seed = imgui.input_int('##diffusion_seed', self.params.seed)
                    imgui.same_line(spacing=0)
                    imgui.text('Seed')

                    _changed, self.params.lora_path = imgui_utils.input_text(
                        '##diffusion_lora', self.params.lora_path, 1024, 0,
                        width=-1 - viz.app.button_w * 2 - viz.app.spacing * 2, help_text="lora file")
                    imgui.same_line()
                    if imgui_utils.button('Find##lora', width=viz.app.button_w):
                        lora = self.browser.select_lora_file(initial_dir=self.params.lora_path)
                        if lora:
                            self.params.lora_path = str(lora)
                    imgui.same_line()
                    with imgui_utils.item_width(viz.app.button_w):
                        _changed, self.params.lora_scale = imgui.slider_float('##diffusion_lora_scale',
                                                                              self.params.lora_scale, 0, 2,
                                                                              format='LoRA %.2f')

                    accel_index = (ACCELERATIONS.index(self.params.acceleration)
                                   if self.params.acceleration in ACCELERATIONS else 0)
                    with imgui_utils.item_width(viz.app.button_w * 2):
                        _changed, accel_index = imgui.combo('##diffusion_acceleration', accel_index,
                                                            ACCELERATION_LABELS)
                    self.params.acceleration = ACCELERATIONS[accel_index]
                    needs_build = self.params.acceleration == 'tensorrt' and not self.engines_ready()
                    imgui.same_line(spacing=viz.app.spacing * 2)
                    if imgui_utils.button('Build engines##diffusion', width=viz.app.button_w * 1.5,
                                          enabled=(self.enabled and needs_build
                                                   and self.build_state == 'idle')):
                        self.start_build()

                    if status:
                        imgui.text_colored(status, 1.0, 0.3, 0.3, 1.0)
                    elif needs_build:
                        imgui.text_colored(diffusion_engine.TRT_NOT_BUILT, 1.0, 0.3, 0.3, 1.0)
                    else:
                        imgui.text('Ready' if self.enabled else 'Off')

            self.osc_menu()

        self.draw_build_modal()
        self.params.strength = float(min(max(self.params.strength, 0.0), 1.0))
        self.params.seed = int(self.params.seed)
        self.params.lora_scale = float(min(max(self.params.lora_scale, 0.0), 2.0))
        if self.params.acceleration not in ACCELERATIONS:
            self.params.acceleration = ACCELERATIONS[0]
        viz.args.use_diffusion = bool(self.enabled and self.available)
        viz.args.diffusion = dict(self.params)

#----------------------------------------------------------------------------
