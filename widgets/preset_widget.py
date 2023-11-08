import os
import imgui
from utils.gui_utils import imgui_utils
from widgets.browse_widget import BrowseWidget



import numpy as np

class PresetWidget:
    def __init__(self, viz):
        self.viz = viz
        self.num_presets = 12
        self.c_inactive = imgui.get_style().colors[imgui.COLOR_MENUBAR_BACKGROUND]
        self.active = np.asarray([False] * self.num_presets)

        self.path = "presets"
        self.tmp_path = self.path
        self.paths = np.asarray([f"./{self.path}/{i}" for i in range(self.num_presets)])
        self.check_presets()
        self.recent_paths = [self.path]
        self.use_osc = False
        self.osc_addresses = ""
        self.browser = BrowseWidget(self.viz, "Preset Path##presetwidget", os.path.abspath(os.getcwd()), [""], width=viz.app.font_size * 10, multiple=False, traverse_folders=False)



    def check_presets(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.assigned = np.ones(self.num_presets)
        for p in os.listdir(self.path):
            self.assigned[int(p)] = 0

    def open_path(self, path):
        try:
            if not os.path.exists(path):
                os.makedirs(path)
            self.paths = np.asarray([f"./{self.path}/{i}" for i in range(self.num_presets)])
            self.path = self.tmp_path
            self.check_presets()
        except Exception as e:
            print(e)

    def save(self, path):
        try:
            if not os.path.exists(path):
                os.makedirs(path)
            # self.inter.terminal_widget.cached_text.append(f"Saving Preset at {path}")
            self.viz.latent_widget.save(f"{path}/latent.pkl")
            self.viz.trunc_noise_widget.save(f"{path}/trunc.pkl")
            self.viz.layer_widget.save(f"{path}/layer.pkl")
            self.viz.adjuster_widget.save(f"{path}/adjuster.pkl")
            self.viz.looping_widget.save(f"{path}/looper.pkl")
            self.viz.pickle_widget.save(f"{path}/pickle.pkl")
            self.assigned[np.where(self.active)] = 0
        except Exception as e:
            print(e)

    def load(self, path):
        try:
            # self.viz.terminal_widget.cached_text.append(f"Loading Preset from {path}")
            self.viz.latent_widget.load(f"{path}/latent.pkl")
            self.viz.trunc_noise_widget.load(f"{path}/trunc.pkl")
            self.viz.layer_widget.load(f"{path}/layer.pkl")
            self.viz.adjuster_widget.load(f"{path}/adjuster.pkl")
            self.viz.looping_widget.load(f"{path}/looper.pkl")
            self.viz.pickle_widget.load(f"{path}/pickle.pkl")
            self.viz.app.skip_frame()
        except Exception as e:
            print(e)

    @imgui_utils.scoped_by_object_id
    def preset_checkbox(self, i):
        with imgui_utils.grayed_out(self.assigned[i]):
            clicked, _ = imgui.checkbox(f"##{i}", self.active[i])
            if clicked:
                self.active *= False
                self.active[i] = True

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz

        if show:
            for i in range(len(self.active)):
                self.preset_checkbox(i)
                imgui.same_line()

            imgui.same_line()
            if imgui_utils.button('Load##presets', width=viz.app.button_w):
                self.load(self.paths[np.where(self.active)].item())

            imgui.same_line()
            if imgui_utils.button("Save##presets", width=viz.app.button_w):
                if self.active.any():
                    self.save(self.paths[np.where(self.active)].item())
                    self.assigned[np.where(self.active)] = 0
            with imgui_utils.item_width(viz.app.button_w * 2), imgui_utils.grayed_out(True):
                imgui.input_text("##preset_path", self.tmp_path, 256, imgui.INPUT_TEXT_READ_ONLY)
            imgui.same_line()
            _clicked, file_out = self.browser(self.viz.app.button_w)

            if _clicked:
                self.tmp_path = file_out[0]

            imgui.same_line()
            if imgui_utils.button('Recent...##presets', width=viz.app.button_w, enabled=(len(self.recent_paths) != 0)):
                imgui.open_popup('recent_preset_popup')
            if imgui.begin_popup('recent_preset_popup'):
                for pth in self.recent_paths:
                    clicked, _state = imgui.menu_item(pth)
                    if clicked:
                        self.open_path(pth)
                imgui.end_popup()
            imgui.same_line()

            if imgui_utils.button('Open##presets', width=viz.app.button_w):
                self.open_path(self.tmp_path)
                if not self.tmp_path in self.recent_paths:
                    self.recent_paths.append(self.tmp_path)
            imgui.same_line()
            _, self.use_osc = imgui.checkbox(f"Use OSC##load", self.use_osc)
            imgui.same_line()
            with imgui_utils.grayed_out(not (self.use_osc)):

                changed, osc_address = imgui_utils.input_text(f"##OSC_load",
                                                              self.osc_addresses,
                                                              256,
                                                              imgui.INPUT_TEXT_CHARS_NO_BLANK |
                                                              (
                                                                  imgui.INPUT_TEXT_READ_ONLY) * (
                                                                  not self.use_osc),
                                                              width=viz.app.font_size * 5,
                                                              help_text="Osc Address")
                if changed:
                    try:
                        viz.osc_dispatcher.unmap(f"/{self.osc_addresses}",
                                                   self.osc_handler)
                        self.osc_addresses = osc_address
                    except:
                        print(f"{self.osc_addresses} is not mapped")
                    viz.osc_dispatcher.map(f"/{self.osc_addresses}",
                                             self.osc_handler)

    def osc_handler(self, address, *args):
        try:
            value = args[-1]
            if value > len(self.active):
                value = len(self.active) - 1
            value = max(value, 0)
            self.active *= False
            self.active[value] = True
            self.load(self.paths[np.where(self.active)].item())
        except Exception as e:
            print(e, "preset")
