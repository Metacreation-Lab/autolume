import datetime
import logging
import os
import re
import threading

import imgui
import glfw

import upscale
from assets.colors import RED
from utils import device_utils
from utils.gui_utils import imgui_utils
from utils.user_data import data_path

logger = logging.getLogger(__name__)

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

        # Live upscaling. The checkbox and combo hold the selection, but only
        # settings whose weights are on disk become the active pass.
        self.upscale_enabled = False
        self.upscale_model = upscale.PERFORM_DEFAULT_MODEL
        self._upscale_labels = [upscale.PERFORM_LABELS[key] for key in upscale.PERFORM_MODELS]
        self._upscale_ready = None
        self._download_thread = None
        self._download_key = None
        self._download_ok = False

    def _upscale_key(self):
        """The model the current selection asks for, or None when off."""
        return self.upscale_model if self.upscale_enabled else None

    def _sync_upscale_weights(self):
        """Activate the selection, or download the weights it still needs."""
        self._upscale_ready = None
        if not self.upscale_enabled or self._download_thread is not None:
            return
        key = self._upscale_key()
        missing = [name for name in upscale.required_weights(key)
                   if not os.path.exists(upscale.weight_path(name))]
        if not missing:
            self._upscale_ready = key
            return
        self._download_key = key
        self._download_ok = False
        self._download_thread = threading.Thread(target=self._download_upscale_weights,
                                                 args=(missing,), daemon=True,
                                                 name='perform-upscale-weights')
        self._download_thread.start()

    def _download_upscale_weights(self, names):
        try:
            for name in names:
                if upscale.ensure_weight(name) is None:
                    return
            self._download_ok = True
        except Exception:
            logger.exception("Live upscaling weight download failed")

    def _poll_upscale_download(self):
        """Pick up a finished download. A failed one falls back to Off."""
        if self._download_thread is None or self._download_thread.is_alive():
            return
        key, ok = self._download_key, self._download_ok
        self._download_thread = None
        self._download_key = None
        self._download_ok = False
        if key != self._upscale_key():
            self._sync_upscale_weights()
        elif ok:
            self._upscale_ready = key
        else:
            self.upscale_enabled = False

    def _capture_basename(self):
        pkl = self.viz.pickle_widget.cur_pkl
        model = os.path.splitext(os.path.basename(pkl))[0] if pkl else 'autolume'
        model = re.sub(r'\s+', '-', model.strip())
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        return f'{model}_{timestamp}'

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        self._poll_upscale_download()
        if show:
            if imgui.radio_button('Raw Scale', not viz.fit_screen):
                viz.fit_screen = False
            imgui.same_line(spacing=viz.app.spacing * 2)
            if imgui.radio_button('Fit', viz.fit_screen):
                viz.fit_screen = True
            imgui.same_line(spacing=viz.app.spacing * 2)
            with imgui_utils.grayed_out(not viz.fit_screen):
                bilinear_clicked, bilinear_new = imgui.checkbox('Bilinear', viz.bilinear_display)
            if bilinear_clicked and viz.fit_screen:
                viz.bilinear_display = bilinear_new

            imgui.same_line(spacing=viz.app.spacing * 2)
            upscale_clicked, upscale_new = imgui.checkbox('Upscale', self.upscale_enabled)
            if upscale_clicked:
                self.upscale_enabled = upscale_new
                self._sync_upscale_weights()
            imgui.same_line()
            with imgui_utils.grayed_out(not self.upscale_enabled):
                with imgui_utils.item_width(viz.app.font_size * 6):
                    model_clicked, model_index = imgui.combo(
                        '##upscale_model', upscale.PERFORM_MODELS.index(self.upscale_model),
                        self._upscale_labels)
            if model_clicked:
                self.upscale_model = upscale.PERFORM_MODELS[model_index]
                self._sync_upscale_weights()

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

            if self._download_thread is not None:
                imgui.same_line(spacing=viz.app.spacing * 2)
                imgui.text('Downloading model')

        viz.args.upscale_model = self._upscale_ready

#----------------------------------------------------------------------------
