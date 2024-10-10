import imgui
import librosa
import numpy as np
import torch
from pythonosc.udp_client import SimpleUDPClient

import dnnlib
from assets import ACTIVE_RED
from audio import audio_stream
from utils.gui_utils import imgui_utils

chromas = ["C", "C/D", "D", "D/E", "E", "E/F", "F", "F/G" "G", "G/A", "A", "A/B", "B"]


class AudioWidget:
    """
    Audio Widget that takes default audio device and performs FFT on signal
    Input audio signal can be separated into harmonic and percussive components
    Signal is sent via OSC to Osc-address (self.osc_addresses) if self.use_osc is True
    """

    def __init__(self, viz):
        self.viz = viz
        self.decompose = False

        self.fft = dnnlib.EasyDict(data=None, min=0, max=0, use_H=False, use_P=False)
        self.n_fft = 512

        self.osc_addresses = dnnlib.EasyDict({"fft": "osc address", "H": "osc address", "P": "osc address"})
        self.mappings = dnnlib.EasyDict(zip(self.osc_addresses.keys(), ["x"] * len(self.osc_addresses)))
        self.use_osc = dnnlib.EasyDict(zip(self.osc_addresses.keys(), [False] * len(self.osc_addresses)))

        self.audio_stream = audio_stream.AudioStream(pa=viz.pa,rate=44100,
                                                     callback=self.callback)  # make audio stream work with threading
        self.audio_stream.stream_start()

    def get_params(self):
        return (self.n_fft, self.osc_addresses, self.mappings, self.use_osc)

    def set_params(self, params):
        self.n_fft, self.osc_addresses, self.mappings, self.use_osc = params

    def callback(self, data):
        self.fft.data = np.abs(librosa.stft(data, n_fft=self.n_fft * 2 - 1))

    @imgui_utils.scoped_by_object_id
    def send_osc(self, key, signal):
        viz = self.viz

        draw_list = imgui.get_window_draw_list()

        draw_list.channels_split(2)
        draw_list.channels_set_current(1)
        _, self.use_osc[key] = imgui.checkbox(f"Use OSC##_{key}", self.use_osc[key])
        if self.use_osc[key]:
            draw_list.channels_set_current(0)
            p_min = imgui.get_item_rect_min()
            p_max = imgui.get_item_rect_max()
            draw_list.add_rect_filled(p_min.x + (self.viz.app.font_size * 1.5), p_min.y,
                                      p_min.x + (self.viz.app.button_w), p_max.y, imgui.get_color_u32_rgba(*ACTIVE_RED))

        draw_list.channels_merge()
        imgui.same_line()
        with imgui_utils.item_width(viz.app.font_size * 5),imgui_utils.grayed_out(not(self.use_osc[key])):
            changed, self.osc_addresses[key] = imgui.input_text(f"##OSC_{key}", self.osc_addresses[key], 256,
                                                             imgui.INPUT_TEXT_CHARS_NO_BLANK | imgui.INPUT_TEXT_ENTER_RETURNS_TRUE | (
                                                                   imgui.INPUT_TEXT_READ_ONLY * (
                                                               not self.use_osc[key])))
            imgui.same_line()
            changed, self.mappings[key] = imgui.input_text(f"##Mappings_{key}",
                                                           self.mappings[key], 256,
                                                           imgui.INPUT_TEXT_ENTER_RETURNS_TRUE | (
                                                                   imgui.INPUT_TEXT_READ_ONLY * (
                                                               not self.use_osc[key])))

        # self.signal_queue.put((signal, self.mappings[key], self.osc_addresses[key], self.use_osc[key]))
        if self.use_osc[key]:
            try:
                f = lambda x: eval(self.mappings[key])
                viz.osc_client.send_message(f"/{self.osc_addresses[key]}", [f(signal).tolist()])
                #print(f"/{self.osc_addresses[key]}", f(signal).tolist())
            except Exception as e:
                print(e)
    @staticmethod
    def osc_process(signal_queue):
        ip, port = "127.0.0.1", 1337
        osc_client = SimpleUDPClient(ip, port)
        while True:
            signal, mapping, address, use_osc, tmp_ip, tmp_port = signal_queue.get()
            if not(tmp_ip == ip or tmp_port==port):
                osc_client = SimpleUDPClient(ip,port)
            try:
                if use_osc:
                    f = lambda x: eval(mapping)
                    osc_client.send_message(address, [f(signal).tolist()])
                    print(address, f(signal).tolist())
            except Exception as e:
                print(e)

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        plot_values= 0
        if show:
            width = viz.app.font_size
            height = imgui.get_text_line_height_with_spacing()
            bg_color = [0.16, 0.29, 0.48, 0.2]
            _, self.decompose = imgui.checkbox("Decompose##extract", self.decompose)
            if self.decompose:
                imgui.begin_group()
                try:
                    H, P = librosa.decompose.hpss(S=self.fft.data)
                    imgui.plot_histogram("##Harmonic", np.median(H, axis=1), graph_size=(width * 24, height * 1.5))
                    imgui.text("Harmonic Signal")
                    imgui.plot_histogram("##Percussive", np.median(P, axis=1),
                                         graph_size=(width * 24, height * 1.5))
                    imgui.text("Percussive Signal")
                except Exception as e:
                    print(e, "Audio Decomp")
                imgui.end_group()
            elif not (self.fft.data is None):
                try:
                    plot_values = np.median(self.fft.data, axis=1)
                    self.fft.min = plot_values.min()
                    self.fft.max = (0.2 * plot_values.max()) + (0.8 * self.fft.max)
                    imgui.plot_histogram("##AudioSignal", plot_values, graph_size=(width * 24, height * 3),
                                         scale_min=self.fft.min, scale_max=self.fft.max)
                except Exception as e:
                    print(e, "AUDIO")

            imgui.same_line()
            imgui.begin_group()
            with imgui_utils.item_width(viz.app.font_size * 10):
                changed, self.n_fft = imgui.input_int("Number of FFT", self.n_fft,
                                                      flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
            if changed:
                if self.n_fft < 3:
                    self.n_fft = 3
                self.audio_stream.callback = self.callback

            if self.decompose:
                self.send_osc("H", np.median(H, axis=1))
                self.send_osc("P", np.median(P, axis=1))
            else:
                self.send_osc("fft", plot_values)
            imgui.end_group()
