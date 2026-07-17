import datetime
import os
import re

import imgui
import glfw

from assets.colors import RED
from utils import device_utils
from utils.gui_utils import imgui_utils
from utils.user_data import data_path

#----------------------------------------------------------------------------

def _toggle_button_width(*labels):
    # Fixed width fitting the widest label, so toggling labels never reflows the row.
    pad = imgui.get_style().frame_padding[0] * 2
    return max(imgui.calc_text_size(label)[0] for label in labels) + pad

#----------------------------------------------------------------------------

class DisplayCaptureWidget:
    def __init__(self, viz):
        self.viz = viz
        # The fullscreen display uses a core-profile GL 3.3 context that shares
        # textures with the main legacy context; macOS cannot share across those
        # profiles, so the button is omitted there.
        self.show_fullscreen = not device_utils.is_macos()
        self.use_superres = False

    def _capture_basename(self):
        pkl = self.viz.pickle_widget.cur_pkl
        model = os.path.splitext(os.path.basename(pkl))[0] if pkl else 'autolume'
        model = re.sub(r'\s+', '-', model.strip())
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        return f'{model}_{timestamp}'

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            if imgui.radio_button('Raw Scale', not viz.fit_screen):
                viz.fit_screen = False
            imgui.same_line(spacing=viz.app.spacing * 2)
            if imgui.radio_button('Fit', viz.fit_screen):
                viz.fit_screen = True
            imgui.same_line(spacing=viz.app.spacing * 2)
            _clicked, self.use_superres = imgui.checkbox('Super Resolution', self.use_superres)

            if self.show_fullscreen:
                imgui.same_line(spacing=viz.app.spacing * 2)
                fullscreen_label = 'Fullscreen' if not viz.is_fullscreen_display else 'Exit Fullscreen'
                if imgui_utils.button(fullscreen_label, width=_toggle_button_width('Fullscreen', 'Exit Fullscreen')):
                    if viz.is_fullscreen_display:
                        viz.is_fullscreen_display = False
                        if viz.fullscreen_window:
                            glfw.destroy_window(viz.fullscreen_window)
                            viz.fullscreen_window = None
                            viz.window_created = False
                    else:
                        viz.is_fullscreen_display = True
                        viz.window_created = False

            imgui.same_line(spacing=viz.app.spacing * 2)
            if imgui.button('Screenshot'):
                viz.capture_screenshot(str(data_path('captures', f'{self._capture_basename()}.png')))
            imgui.same_line(spacing=viz.app.spacing * 2)
            record_w = _toggle_button_width('Record', 'Stop')
            if viz.is_recording:
                clicked_record = imgui_utils.color_button('Stop', RED, width=record_w)
            else:
                clicked_record = imgui_utils.button('Record', width=record_w)
            if clicked_record:
                if not viz.is_recording:
                    viz.start_recording(str(data_path('captures', f'{self._capture_basename()}.mp4')))
                else:
                    viz.stop_recording()

        viz.args.use_superres = self.use_superres

#----------------------------------------------------------------------------
