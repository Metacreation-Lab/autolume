import logging
import imgui

import dnnlib
from diffusion import engine as diffusion_engine
from utils.gui_utils import imgui_utils
from widgets import osc_menu

logger = logging.getLogger(__name__)

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

MODELS = ["stabilityai/sd-turbo", "KBlueLeaf/kohaku-v2.1"]

#----------------------------------------------------------------------------

class DiffusionWidget:
    def __init__(self, viz):
        self.viz = viz
        self.available = diffusion_engine.is_available()
        self.enabled = False
        self.params = dnnlib.EasyDict(diffusion_engine.default_params())
        self.custom_model = ""
        self.model_index = 0

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

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        status = viz.result.get('diffusion_status', '')

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

                    if status:
                        imgui.text_colored(status, 1.0, 0.3, 0.3, 1.0)
                    else:
                        imgui.text('Ready' if self.enabled else 'Off')

            self.osc_menu()

        self.params.strength = float(min(max(self.params.strength, 0.0), 1.0))
        self.params.seed = int(self.params.seed)
        viz.args.use_diffusion = bool(self.enabled and self.available)
        viz.args.diffusion = dict(self.params)

#----------------------------------------------------------------------------
