import copy
import os

import imgui
import numpy as np
import torch
from opensimplex import OpenSimplex

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

import dnnlib
from utils.gui_utils import imgui_utils
from widgets import osc_menu
from widgets.browse_widget import BrowseWidget

from pythonosc.udp_client import SimpleUDPClient

import multiprocessing as mp


def valmap(value, istart, istop, ostart, ostop):
    return ostart + (ostop - ostart) * ((value - istart) / (istop - istart))


class OSN():
    min = -1
    max = 1

    def __init__(self, seed, diameter):
        self.tmp = OpenSimplex(seed)
        self.d = diameter
        self.x = 0
        self.y = 0

    def get_val(self, angle):
        self.xoff = valmap(np.cos(angle), -1, 1, self.x, self.x + self.d)
        self.yoff = valmap(np.sin(angle), -1, 1, self.y, self.y + self.d)
        return self.tmp.noise2(self.xoff, self.yoff)


def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    '''
    Spherical linear vizpolation
    Args:
        t (float/np.ndarray): Float value between 0.0 and 1.0
        v0 (np.ndarray): Starting vector
        v1 (np.ndarray): Final vector
        DOT_THRESHOLD (float): Threshold for considering the two vectors as
                               colineal. Not recommended to alter this.
    Returns:
        v2 (np.ndarray): vizpolation vector between v0 and v1
    '''
    v0 = v0.cpu().detach().numpy()
    v1 = v1.cpu().detach().numpy()
    # Copy the vectors to reuse them later
    v0_copy = np.copy(v0)
    v1_copy = np.copy(v1)
    # Normalize the vectors to get the directions and angles
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    # Dot product with the normalized vectors (can't use np.dot in W)
    dot = np.sum(v0 * v1)
    # If absolute value of dot product is almost 1, vectors are ~colineal, so use lerp
    if np.abs(dot) > DOT_THRESHOLD:
        return torch.from_numpy(v0_copy + (v1_copy - v0_copy) * t)
    # Calculate initial angle between v0 and v1
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    # Angle at timestep t
    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)
    # Finish the slerp algorithm
    s0 = np.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    v2 = s0 * v0_copy + s1 * v1_copy
    return torch.from_numpy(v2)


labels = ["Seed", "Vector", "Keyframe"]

def noise_loop(args_queue, results_queue):
    while True:
        args = args_queue.get()
        while args_queue.qsize() > 0:
            args = args_queue.get()
        seed, radius = args
        feats = [OSN(seed + i, radius) for i in range(512)]
        results_queue.put(feats)


class LoopingWidget:
    def __init__(self, viz):
        self.params = dnnlib.EasyDict(num_keyframes=6, mode=True, anim=False, index=0, looptime=4, perfect_loop=False)
        self.use_osc = dnnlib.EasyDict(zip(self.params.keys(), [False] * len(self.params)))
        self.step_y = 100
        self.viz = viz
        self.keyframes = [torch.randn(1, 512) for _ in range(self.params.num_keyframes)]
        self.alpha = 0
        self.speed = 0
        self.expand_vec = False
        self.seeds = [[i, 0] for i in range(self.params.num_keyframes)]
        self.modes = [0] * self.params.num_keyframes
        self.project = [True] * self.params.num_keyframes
        self.paths = [""] * self.params.num_keyframes
        self._pinned_bufs = dict()
        self.halt_update = 0
        self.perfect_loop = False
        self.looping_snaps = [{} for _ in range(self.params.num_keyframes)]
        self.file_dialogs = [BrowseWidget(viz, f"Browse##vec{i}", os.path.abspath(os.getcwd()), ["*", ".pth", ".pt"],
                                          width=self.viz.app.button_w, multiple=False, traverse_folders=False) for i in
                             range(self.params.num_keyframes)]
        self.open_keyframes = False
        self.open_file_dialog = False
        self.osc_ip = "127.0.0.1"
        self.osc_port = 5005
        self.osc_client = SimpleUDPClient(self.osc_ip, self.osc_port)
        self.osc_address = ""
        self.looped = 0
        self.loop_type = True
        self.z = torch.randn(1, 512)
        self.radius = 1
        self.noise_seed = 0
        self.noise_loop_feats = [OSN(self.noise_seed + i, self.radius) for i in range(512)]
        self.args_queue = mp.Queue()
        self.results_queue = mp.Queue()
        self.noise_loop_process = mp.Process(target=noise_loop, args=(self.args_queue, self.results_queue), daemon=True)
        self.noise_loop_process.start()

        # flag that tells us we need to stop loop necessary for reverse looping
        self.stop_loop = False

        funcs = dict(zip(["Animate", "Number of Keyframes", "Time", "Index", "Perfect Loop"], [self.osc_handler(param) for param in
                                                                                  ["anim", "num_keyframes", "looptime",
                                                                                   "index", "perfect_loop"]]))
        funcs["Alpha"] = self.alpha_handler()

        funcs["Perfect Loop"] = self.perfect_loop_handler()

        self.time_osc_menu = osc_menu.OscMenu(self.viz, copy.deepcopy(funcs), None,
                                         label="##LoopingTimeOSC")

        del funcs["Time"]

        funcs["Speed"] = self.speed_handler()
        self.speed_osc_menu = osc_menu.OscMenu(self.viz, copy.deepcopy(funcs), None,
                                            label="##LoopingSpeedOSC")
        del funcs["Number of Keyframes"]
        del funcs["Index"]

        funcs["Radius"] = self.radius_handler()
        self.speed_noise_osc_menu = osc_menu.OscMenu(self.viz, copy.deepcopy(funcs), None,
                                            label="##LoopingSpeedNoiseOSC")

        del funcs["Speed"]
        funcs["Time"] = self.osc_handler("looptime")
        self.time_noise_osc_menu = osc_menu.OscMenu(self.viz, copy.deepcopy(funcs), None,
                                            label="##LoopingTimeNoiseOSC")

        self.remove_entry = -1

    def alpha_handler(self):
        def func(address, *args):
            try:
                assert (type(args[-1]) is type(
                    self.alpha)), f"OSC Message and Parameter type must align [OSC] {type(args[-1])} != [Param] {type(self.alpha)}"
                self.alpha = args[-1]
                print(self.alpha, args[-1])
                self.update_alpha()

            except Exception as e:
                self.viz.print_error(e)

        return func

    def osc_handler(self, param):
        def func(address, *args):
            try:
                nec_type = type(self.params[param])
                self.use_osc[param] = True
                self.params[param] = nec_type(args[-1])
            except Exception as e:
                self.viz.print_error(e)

        return func

    # Restructure code to make super class that has save and load
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.get_params(), f)

    def load(self, path):
        with open(path, "rb") as f:
            self.set_params(pickle.load(f))

    def get_params(self):
        return self.params.num_keyframes, self.keyframes, self.alpha, self.params.index, self.params.mode, self.params.anim, self.params.looptime, \
            self.expand_vec, self.seeds, self.modes, self.project, self.paths

    def set_params(self, params):
        self.params.num_keyframes, self.keyframes, self.alpha, self.params.index, self.params.mode, self.params.anim, self.params.looptime, self.expand_vec, self.seeds, self.modes, self.project, self.paths = params

    def drag(self, idx, dx, dy):
        viz = self.viz
        self.seeds[idx][0] += dx / viz.app.font_size * 4e-2
        self.seeds[idx][1] += dy / viz.app.font_size * 4e-2

    def _get_pinned_buf(self, ref):
        key = (tuple(ref.shape), ref.dtype)
        buf = self._pinned_bufs.get(key, None)
        if buf is None:
            buf = torch.empty(ref.shape, dtype=ref.dtype).pin_memory()
            self._pinned_bufs[key] = buf
        return buf

    def to_cpu(self, buf):
        return self._get_pinned_buf(buf).copy_(buf).clone()

    def update_alpha(self):
        step_size = 0
        if self.params.mode:
            if not (self.viz.app._fps_limit is None) and self.params.looptime != 0:
                step_size = (self.params.num_keyframes * self.viz.app.frame_delta) / self.params.looptime

            if self.loop_type is False:
                step_size = (self.viz.app.frame_delta) / self.params.looptime
        else:
            step_size = 0.01 * self.speed

        self.alpha += step_size

        if self.alpha >= 1:
            if self.halt_update < 0:
                print(self.params.index, self.params.num_keyframes, self.loop_type)
                if self.params.index == (self.params.num_keyframes - 1) or self.loop_type is False:
                    self.looped = 1
                    if self.perfect_loop:
                        self.params.anim = False
                if self.loop_type:
                    self.params.index = int(self.params.index + self.alpha) % self.params.num_keyframes
                self.alpha = 0
                self.halt_update = 10
        if step_size < 0:
            if self.alpha <= 0:
                if self.halt_update < 0:
                    if self.stop_loop or self.loop_type is False:
                        self.stop_loop = False
                        self.looped = 1
                        if self.perfect_loop:
                            self.params.anim = False
                    if self.params.index == 1 or len(self.keyframes) == 1:
                        self.stop_loop = True
                    self.params.index = self.params.index + self.alpha
                    if self.params.index <= 0:
                        self.params.index += self.params.num_keyframes
                    if self.loop_type:
                        self.params.index = int(self.params.index % self.params.num_keyframes)
                    self.alpha = 1
                    self.halt_update = 10

        self.halt_update -= 1

    @imgui_utils.scoped_by_object_id
    def key_frame_vizface(self, idx):
        label = labels[self.modes[idx]]
        if imgui_utils.button(f"{label}##{idx}", width=(self.viz.app.font_size * len(label)) / 2):
            self.modes[idx] = (self.modes[idx] + 1) % len(labels) if self.looping_snaps[idx] != {} else (self.modes[
                                                                                                             idx] + 1) % (
                                                                                                                len(labels) - 1)
        imgui.same_line()
        _clicked, self.project[idx] = imgui.checkbox(f'Project##loop{idx}', self.project[idx])
        imgui.same_line()
        imgui.text("|")
        imgui.same_line()
        if self.modes[idx] == 0:
            self.seed_viz(idx)
        elif self.modes[idx] == 1:
            self.vec_viz(idx)
        elif self.modes[idx] == 2:
            imgui.text("Not Implemented")
            imgui.same_line()
            if imgui_utils.button(f"Remove##vecmode{idx}", width=self.viz.app.button_w):
                self.remove_entry = idx

    def open_vec(self, idx):
        try:
            print(self.paths[idx])
            # if file ends with pt or pth, load as torch tensor else if ends with npy, load as numpy array and convert to torch tensor
            if self.paths[idx].endswith(".pt") or self.paths[idx].endswith(".pth"):
                vec = torch.load(self.paths[idx]).squeeze()
            elif self.paths[idx].endswith(".npy"):
                vec = torch.from_numpy(np.load(self.paths[idx])).squeeze()
            else:
                raise Exception(
                    "Filetype not supported, please use .pt, .pth or .npy files for loading vectors, if you are using a .npy file, please ensure that it is a numpy array")
            assert vec.shape[-1] == self.keyframes[
                idx].shape[-1], f"The Tensor you are loading has a different shape, Loaded Shape {vec.shape} != Target Shape {self.keyframes[idx].shape}"
            print("LOADed VEC", vec.shape, torch.unique(vec))
            self.keyframes[idx] = vec
        except Exception as e:
            print(e)

    def vec_viz(self, idx):
        viz = self.viz
        changed, self.paths[idx] = imgui_utils.input_text(f"##vec_path_loop{idx}", self.paths[idx], 256,
                                                          imgui.INPUT_TEXT_CHARS_NO_BLANK,
                                                          width=viz.app.font_size * 7, help_text="filepath")
        imgui.same_line()
        _clicked, path = self.file_dialogs[idx](self.viz.app.button_w)
        if _clicked:
            self.paths[idx] = path[0]
        imgui.same_line()
        if imgui_utils.button(f"Load Vec##loop_{idx}", viz.app.button_w):
            self.open_vec(idx)
        imgui.same_line()
        if imgui_utils.button(f"Snap##{idx}", viz.app.button_w):
            snapped = self.snap()

            if not (snapped is None):
                if snapped["mode"] == 0:
                    self.seeds[idx] = snapped["snap"]
                    self.modes[idx] = snapped["mode"]
                elif snapped["mode"] == 1:
                    self.keyframes[idx] = snapped["snap"]
                    self.modes[idx] = snapped["mode"]
                elif snapped["mode"] == 2:
                    self.looping_snaps[idx] = snapped["snap"]
                    self.modes[idx] = snapped["mode"]
                    print(self.looping_snaps[idx])
                    print(self.modes[idx])

        imgui.same_line()
        if imgui_utils.button(f"Randomize##vecmode{idx}", width=viz.app.button_w):
            self.keyframes[idx] = torch.randn(self.keyframes[idx].shape)

        imgui.same_line()
        if imgui_utils.button(f"Remove##vecmode{idx}", width=viz.app.button_w):
            self.remove_entry = idx

    @imgui_utils.scoped_by_object_id
    def seed_viz(self, idx):
        update_vec = False
        viz = self.viz
        seed = round(self.seeds[idx][0]) + round(self.seeds[idx][1]) * self.step_y
        with imgui_utils.item_width(viz.app.font_size * 8):
            _changed, seed = imgui.input_int(f"##loopseed{idx})", seed)
        if _changed:
            self.seeds[idx][0] = seed
            self.seeds[idx][1] = 0
        imgui.same_line()
        frac_x = self.seeds[idx][0] - round(self.seeds[idx][0])
        frac_y = self.seeds[idx][1] - round(self.seeds[idx][1])
        with imgui_utils.item_width(viz.app.font_size * 5):
            _changed, (new_frac_x, new_frac_y) = imgui.input_float2(f'##loopfrac{idx}', frac_x, frac_y, format='%+.2f',
                                                                    flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
        if _changed:
            self.seeds[idx][0] += new_frac_x - frac_x
            self.seeds[idx][1] += new_frac_y - frac_y

        imgui.same_line()
        _clicked, dragging, dx, dy = imgui_utils.drag_button(f'Drag##loopdrag{idx}', width=viz.app.button_w)
        if dragging:
            self.drag(idx, dx, dy)

        imgui.same_line()
        if imgui_utils.button(f"Snap##seed{idx}", viz.app.button_w):
            snapped = self.snap()
            print("snapped", snapped, "-----------------------")

            if not (snapped is None):
                if snapped["mode"] == 0:
                    print("SEED")
                    self.seeds[idx] = snapped["snap"]  # [snapped["snap"]["x"],snapped["snap"]["y"]]
                    self.modes[idx] = snapped["mode"]
                elif snapped["mode"] == 1:
                    print("VECTOR")
                    self.keyframes[idx] = snapped["snap"]
                    self.modes[idx] = snapped["mode"]
                elif snapped["mode"] == 2:
                    print("getting LOOP")
                    self.looping_snaps[idx] = snapped["snap"]
                    self.modes[idx] = snapped["mode"]
                    print(self.looping_snaps[idx])
                    print(self.modes[idx])
        imgui.same_line()
        if imgui_utils.button(f"Remove##vecmode{idx}", width=viz.app.button_w):
            self.remove_entry = idx

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        if self.results_queue.qsize() > 0:
            self.noise_loop_feats = self.results_queue.get()

        if self.osc_address != "":
            try:
                self.osc_client.send_message(self.osc_address, self.looped)
            except Exception as e:
                print(e)
        viz = self.viz
        # viz.args.looping = self.params.anim

        if show:
            _clicked, self.params.anim = imgui.checkbox('Loop', self.params.anim)
            if _clicked and self.params.anim:
                self.looped = 2
            label = "Keyframe" if self.loop_type else "NoiseLoop"
            imgui.same_line()
            if imgui_utils.button(label, viz.app.button_w):
                self.loop_type = not self.loop_type

            imgui.same_line()
            label = "Time" if self.params.mode else "Speed"
            if imgui_utils.button(f'{label}##loopmode', width=viz.app.button_w, enabled=True):
                self.params.mode = not self.params.mode
            imgui.same_line()
            if self.params.mode:
                imgui.same_line()
                with imgui_utils.item_width(viz.app.font_size * 5):
                    changed, self.params.looptime = imgui.input_int("Seconds", self.params.looptime)
            else:
                with imgui_utils.item_width(viz.app.button_w * 2 - viz.app.spacing * 2):
                    changed, speed = imgui.slider_float('##speed', self.speed, -5, 5, format='Speed %.3f',
                                                        power=3)
                    if changed:
                        self.speed = speed

            if self.loop_type:
                imgui.same_line()
                with imgui_utils.item_width(viz.app.font_size * 5):
                    changed, idx = imgui.input_int("index", self.params.index + 1)
                    if changed:
                        self.params.index = int((idx - 1) % self.params.num_keyframes)
                imgui.same_line()
                with imgui_utils.item_width(viz.app.font_size * 5):
                    changed, self.alpha = imgui.slider_float("alpha", self.alpha, 0, 1)

                with imgui_utils.item_width(viz.app.font_size * 5):
                    changed, new_keyframes = imgui.input_int("# of Keyframes", self.params.num_keyframes)
                if changed and new_keyframes > 0:
                    vecs = [torch.randn(1, 512).cuda() for _ in range(new_keyframes)]
                    vecs[:min(new_keyframes, self.params.num_keyframes)] = self.keyframes[:min(new_keyframes,
                                                                                               self.params.num_keyframes)]
                    self.keyframes = vecs
                    if not self.use_osc:
                        self.params.index = min(self.params.num_keyframes - 2, self.params.index)
                    seeds = [(i, 0) for i in range(new_keyframes)]
                    seeds[:min(new_keyframes, self.params.num_keyframes)] = self.seeds[:min(new_keyframes,
                                                                                            self.params.num_keyframes)]
                    self.seeds = seeds
                    paths = [""] * new_keyframes
                    paths[:min(new_keyframes, self.params.num_keyframes)] = self.paths[:min(new_keyframes,
                                                                                            self.params.num_keyframes)]
                    self.paths = paths
                    modes = [False] * new_keyframes
                    modes[:min(new_keyframes, self.params.num_keyframes)] = self.modes[:min(new_keyframes,
                                                                                            self.params.num_keyframes)]
                    self.modes = modes
                    project = [True] * new_keyframes
                    project[:min(new_keyframes, self.params.num_keyframes)] = self.project[:min(new_keyframes,
                                                                                                self.params.num_keyframes)]
                    self.project = project
                    looping_snaps = [{} for _ in range(new_keyframes)]
                    print("empty looping_snaps", len(looping_snaps))
                    looping_snaps[:min(new_keyframes, self.params.num_keyframes)] = self.looping_snaps[
                                                                                    :min(new_keyframes,
                                                                                         self.params.num_keyframes)]
                    print(looping_snaps)
                    self.looping_snaps = looping_snaps
                    print(self.looping_snaps, looping_snaps)
                    print("Looping snaps add", len(self.looping_snaps), self.params.num_keyframes)

                    file_dialogs = [
                        BrowseWidget(viz, f"Vector##vec{i}", os.path.abspath(os.getcwd()), ["*", ".pth", ".pt"],
                                     width=self.viz.app.button_w, multiple=False, traverse_folders=False) for i in
                        range(new_keyframes)]
                    file_dialogs[:min(new_keyframes, self.params.num_keyframes)] = self.file_dialogs[:min(new_keyframes,
                                                                                                          self.params.num_keyframes)]
                    self.file_dialogs = file_dialogs
                    self.params.num_keyframes = new_keyframes

                imgui.same_line()
                if imgui_utils.button("KeyFrames", width=viz.app.button_w):
                    self.open_keyframes = True
                if self.open_keyframes:
                    collapsed, self.open_keyframes = imgui.begin("KeyFrames", closable=True,
                                                                 flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE | imgui.WINDOW_NO_COLLAPSE)
                    if collapsed:
                        for i in range(self.params.num_keyframes):
                            self.key_frame_vizface(i)
                        if self.remove_entry != -1:
                            del self.keyframes[self.remove_entry]
                            del self.paths[self.remove_entry]
                            del self.project[self.remove_entry]
                            del self.modes[self.remove_entry]
                            del self.seeds[self.remove_entry]
                            del self.looping_snaps[self.remove_entry]
                            del self.file_dialogs[self.remove_entry]

                            print("Looping snaps del", len(self.looping_snaps), self.params.num_keyframes)
                            self.params.num_keyframes -= 1
                            self.remove_entry = -1

                    # check if any file dialogs are open
                    open_dialog = False
                    for file_dialog in self.file_dialogs:
                        if file_dialog.open:
                            open_dialog = True
                            break
                    if self.open_file_dialog == True and not open_dialog:
                        self.open_file_dialog = False
                        imgui.set_window_focus()

                    self.open_file_dialog = open_dialog
                    # check if current window focussed if not close it unless a file dialog is open
                    if not imgui.is_window_focused() and not self.open_file_dialog:
                        self.open_keyframes = False
                    imgui.end()

            else:
                imgui.same_line()
                with imgui_utils.item_width(viz.app.font_size * 5):
                    changed, self.alpha = imgui.slider_float("alpha", self.alpha, 0, 1)
                with imgui_utils.item_width(viz.app.font_size * 5):
                    seed_changed, self.noise_seed = imgui.input_int("Looping Seed", self.noise_seed)

                imgui.same_line()
                with imgui_utils.item_width(viz.app.font_size * 5):
                    radius_changed, self.radius = imgui.input_float("Radius", self.radius)
                if seed_changed or radius_changed:
                    self.args_queue.put((self.noise_seed, self.radius))

            _, self.perfect_loop = imgui.checkbox("Perfect Loop", self.perfect_loop)
            imgui.same_line()
            _changed, self.osc_ip = imgui_utils.input_text("OSC IP", self.osc_ip, 256, imgui.INPUT_TEXT_CHARS_NO_BLANK,
                                                           width=viz.app.font_size * 4.5)
            if _changed:
                self.osc_client = SimpleUDPClient(self.osc_ip, self.osc_port)
            imgui.same_line(spacing=viz.app.spacing* 2)
            with imgui_utils.item_width(viz.app.font_size * 6):
                _changed, self.osc_port = imgui.input_int("OSC Port", self.osc_port)
            if _changed:
                self.osc_client = SimpleUDPClient(self.osc_ip, self.osc_port)

            imgui.same_line(spacing=viz.app.spacing * 2)
            _changed, self.osc_address = imgui_utils.input_text("OSC Address", self.osc_address, 256,
                                                                imgui.INPUT_TEXT_CHARS_NO_BLANK,
                                                                width=viz.app.font_size * 7)

            imgui.same_line()
            with imgui_utils.grayed_out(True):
                imgui.checkbox("Looped", self.looped == 1)
                self.looped = 0
            if self.loop_type:
                if self.params.mode:
                    self.time_osc_menu()
                else:
                    self.speed_osc_menu()
            else:
                if self.params.mode:
                    self.time_noise_osc_menu()
                else:
                    self.speed_noise_osc_menu()

            if self.params.anim:
                self.update_alpha()
                viz.args.alpha = self.alpha
                if self.loop_type:
                    viz.args.looping_index = self.params.index

                    viz.args.mode = "loop"
                    l_list = []
                    for i, mode in enumerate(self.modes):
                        if mode == 0:
                            l_list.append({"mode": "seed", "latent": self.seeds[i], "project": self.project[i]})
                        elif mode == 1:
                            l_list.append({"mode": "vec", "latent": self.keyframes[i], "project": self.project[i]})
                        elif mode == 2:
                            print("adding loop", self.looping_snaps[i])
                            l_list.append({"mode": "loop", "looping_list": self.looping_snaps[i]["looping_list"],
                                           "looping_index": self.looping_snaps[i]["index"],
                                           "alpha": self.looping_snaps[i]["alpha"]})
                    viz.args.looping_list = l_list
                else:
                    viz.args.mode = "vec"
                    for i in range(512):
                        self.z[0, i] = self.noise_loop_feats[i].get_val((np.pi * 2) * self.alpha)
                    viz.args.vec = self.z


    def snap(self):
        return self.viz.result.get("snap", None)

    def speed_handler(self):
        def func(address, *args):
            try:
                self.speed = float(args[-1])
                self.update = True
            except Exception as e:
                self.viz.print_error(e)
        return func

    def radius_handler(self):
        def func(address, *args):
            try:
                self.radius = float(args[-1])
                self.update = True
            except Exception as e:
                self.viz.print_error(e)
        return func
    
    def perfect_loop_handler(self):
        def func(address, *args):
            try:
                self.perfect_loop = bool(args[-1])
                self.update = True
            except Exception as e:
                self.viz.print_error(e)
        return func

