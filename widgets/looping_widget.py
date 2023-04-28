
import imgui
import numpy as np
import torch
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

import dnnlib
from utils.gui_utils import imgui_utils
from widgets import osc_menu



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





class LoopingWidget:
    def __init__(self, viz):
        self.params = dnnlib.EasyDict(num_keyframes=6, mode=True, anim=False, index=0, looptime=4)
        self.use_osc = dnnlib.EasyDict(zip(self.params.keys(), [False] * len(self.params)))
        self.step_y = 100
        self.viz = viz
        self.keyframes = torch.randn(self.params.num_keyframes, 512)
        self.alpha = 0
        self.speed = 0
        self.expand_vec = False
        self.seeds = [dnnlib.EasyDict(x=i,y=0) for i in range(self.params.num_keyframes)]
        self.modes = [False] * self.params.num_keyframes
        self.project = [True]*self.params.num_keyframes
        self.paths = [""] * self.params.num_keyframes
        self._pinned_bufs = dict()
        self._device = torch.device('cuda')
        self.halt_update = 0
        self.perfect_loop = False



        funcs = dict(zip(["anim", "num_keyframes", "looptime", "index", "mode"], [self.osc_handler(param) for param in
                                                                                         ["anim", "num_keyframes", "looptime", "index", "mode"]]))

        funcs["alpha"] = self.alpha_handler()

        self.osc_menu = osc_menu.OscMenu(self.viz, funcs, None,
                                         label="##LoopingOSC")


    def alpha_handler(self):
        def func(address, *args):
            try:
                assert (type(args[-1]) is type(self.alpha)), f"OSC Message and Parameter type must align [OSC] {type(args[-1])} != [Param] {type(self.alpha)}"
                self.alpha = args[-1]
                print(self.alpha, args[-1])
                self.update_alpha()

            except Exception as e:
                self.viz.print_error(e)
        return func
    def osc_handler(self, param):
        def func(address, *args):
            try:
                assert (type(args[-1]) is type(self.params[
                                                   param])), f"OSC Message and Parameter type must align [OSC] {type(args[-1])} != [Param] {type(self.params[param])}"
                self.use_osc[param] = True
                self.params[param] = args[-1]
                print(self.params[param], args[-1])
            except Exception as e:
                self.viz.print_error(e)

        return func

    #Restructure code to make super class that has save and load
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.get_params(), f)

    def load(self, path):
        with open(path, "rb") as f:
            self.set_params(pickle.load(f))




    def get_params(self):
        return self.params.num_keyframes, self.keyframes, self.alpha, self.params.index, self.params.mode, self.params.anim, self.params.looptime,\
               self.expand_vec,self.seeds, self.modes, self.project, self.paths

    def set_params(self, params):
        self.params.num_keyframes, self.keyframes, self.alpha, self.params.index, self.params.mode, self.params.anim, self.params.looptime, self.expand_vec,self.seeds, self.modes, self.project, self.paths = params

    def drag(self,idx, dx, dy):
        viz = self.viz
        self.seeds[idx].x += dx / viz.app.font_size * 4e-2
        self.seeds[idx].y += dy / viz.app.font_size * 4e-2

    def _get_pinned_buf(self, ref):
        key = (tuple(ref.shape), ref.dtype)
        buf = self._pinned_bufs.get(key, None)
        if buf is None:
            buf = torch.empty(ref.shape, dtype=ref.dtype).pin_memory()
            self._pinned_bufs[key] = buf
        return buf

    def to_device(self, buf):
        return self._get_pinned_buf(buf).copy_(buf).to(self._device)

    def to_cpu(self, buf):
        return self._get_pinned_buf(buf).copy_(buf).clone()

    def update_alpha(self):
        step_size = 0
        if self.params.mode:
            if not(self.viz.app._fps_limit is None)and self.params.looptime!=0:
                step_size = (self.params.num_keyframes / self.viz.app._fps_limit)/self.params.looptime
        else:
            step_size = 0.01 * self.speed
        self.alpha += step_size
        if self.alpha >= 1:
            if self.halt_update < 0:
                self.params.index = int(self.params.index+self.alpha)%self.params.num_keyframes
                self.alpha = 0
                if self.params.index == 0 and self.perfect_loop:
                    self.params.anim = False
            self.halt_update = 10
        if step_size < 0:
            if self.alpha <= 0:
                if self.halt_update < 0:
                    self.params.index = self.params.index+self.alpha
                    if self.params.index <= 0:
                        self.params.index += self.params.num_keyframes
                    self.params.index = int(self.params.index %self.params.num_keyframes)
                    self.alpha = 1
                    if self.params.index == 0 and self.perfect_loop:
                        self.params.anim = False
                self.halt_update = 10


        self.halt_update -= 1
        print(self.halt_update)

    @imgui_utils.scoped_by_object_id
    def key_frame_vizface(self, idx):
        label = "Seed" if self.modes[idx] else "Vector"
        if imgui_utils.button(f"{label}##{idx}", width=(self.viz.app.font_size*len(label))/2):
            self.modes[idx] = not self.modes[idx]
        imgui.same_line()
        _clicked, self.project[idx] = imgui.checkbox(f'Project##loop{idx}', self.project[idx])
        imgui.same_line()
        imgui.text("|")
        imgui.same_line()
        if self.modes[idx]:
            self.seed_viz(idx)
        else:
            self.vec_viz(idx)

    def open_vec(self, idx):
        try:
            vec = torch.load(self.paths[idx]).squeeze()
            assert vec.shape == self.keyframes[idx].shape, f"The Tensor you are loading has a different shape, Loaded Shape {vec.shape} != Target Shape {self.keyframes[idx].shape}"
            self.keyframes[idx] = vec
        except Exception as e:
            print(e)

    def vec_viz(self, idx):
        viz = self.viz
        changed, self.paths[idx] = imgui_utils.input_text(f"##vec_path_loop{idx}", self.paths[idx], 256,
                                                            imgui.INPUT_TEXT_CHARS_NO_BLANK,
                                                            width=viz.app.font_size*7, help_text="filepath")
        imgui.same_line()
        if imgui_utils.button(f"Load Vec##loop_{idx}", viz.app.button_w):
            self.open_vec(idx)
        imgui.same_line()
        if imgui_utils.button(f"Snap##{idx}", viz.app.button_w):
            snapped = self.snap()
            if not(snapped is None):
                print("snapped", idx, snapped.shape)
                self.keyframes[idx] = snapped
        imgui.same_line()
        if imgui_utils.button(f"Randomize##vecmode{idx}", width=viz.app.button_w):
            self.keyframes[idx] = torch.randn(self.keyframes[idx].shape)


    @imgui_utils.scoped_by_object_id
    def seed_viz(self, idx):
        update_vec = False
        viz = self.viz
        seed = round(self.seeds[idx].x) + round(self.seeds[idx].y) * self.step_y
        with imgui_utils.item_width(viz.app.font_size * 8):
            _changed, seed = imgui.input_int(f"##loopseed{idx})", seed)
        if _changed:
            self.seeds[idx].x = seed
            self.seeds[idx].y = 0
        imgui.same_line()
        frac_x = self.seeds[idx].x - round(self.seeds[idx].x)
        frac_y = self.seeds[idx].y - round(self.seeds[idx].y)
        with imgui_utils.item_width(viz.app.font_size * 5):
            _changed, (new_frac_x, new_frac_y) = imgui.input_float2(f'##loopfrac{idx}', frac_x, frac_y, format='%+.2f',
                                                                    flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
        if _changed:
            self.seeds[idx].x += new_frac_x - frac_x
            self.seeds[idx].y += new_frac_y - frac_y

        imgui.same_line()
        _clicked, dragging, dx, dy = imgui_utils.drag_button(f'Drag##loopdrag{idx}', width=viz.app.button_w)
        if dragging:
            self.drag(idx,dx, dy)

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        viz.args.looping = self.params.anim

        if show:
            _clicked, self.params.anim = imgui.checkbox('AnimLoop', self.params.anim)
            with imgui_utils.item_width(viz.app.font_size*5):
                changed, new_keyframes = imgui.input_int("# of Keyframes", self.params.num_keyframes)
            if changed and new_keyframes > 0:
                vecs= torch.randn(new_keyframes,512)
                vecs[:min(new_keyframes,self.params.num_keyframes)] = self.keyframes[:min(new_keyframes,self.params.num_keyframes)]
                self.keyframes = vecs
                if not self.use_osc:
                    self.params.index = min(self.params.num_keyframes-2, self.params.index)
                seeds = [dnnlib.EasyDict(x=i,y=0) for i in range(new_keyframes)]
                seeds[:min(new_keyframes, self.params.num_keyframes)] = self.seeds[:min(new_keyframes, self.params.num_keyframes)]
                self.seeds = seeds
                paths = [""]*new_keyframes
                paths[:min(new_keyframes, self.params.num_keyframes)] = self.paths[:min(new_keyframes, self.params.num_keyframes)]
                self.paths = paths
                modes = [False]*new_keyframes
                modes[:min(new_keyframes, self.params.num_keyframes)] = self.modes[:min(new_keyframes, self.params.num_keyframes)]
                self.modes = modes
                project = [True]*new_keyframes
                project[:min(new_keyframes, self.params.num_keyframes)] = self.project[:min(new_keyframes, self.params.num_keyframes)]
                self.project = project
                self.params.num_keyframes = new_keyframes
            imgui.same_line()
            label = "Time" if self.params.mode else "Speed"
            if imgui_utils.button(f'{label}##loopmode', width=viz.app.button_w,enabled=True):
                self.params.mode = not self.params.mode
            imgui.same_line()
            if self.params.mode:
                imgui.same_line()
                with imgui_utils.item_width(viz.app.font_size * 5):
                    changed, self.params.looptime = imgui.input_int("Time to loop", self.params.looptime)
            else:
                with imgui_utils.item_width(viz.app.button_w * 2 - viz.app.spacing * 2):
                    changed, speed = imgui.slider_float('##speed', self.speed, -5, 5, format='Speed %.3f',
                                                        power=3)
                    if changed:
                        self.speed = speed
            imgui.same_line()
            opened = imgui_utils.popup_button("KeyFrames", width=viz.app.button_w)
            if opened:
                for i in range(self.params.num_keyframes):
                    self.key_frame_vizface(i)
                imgui.end_popup()
            imgui.same_line()
            with imgui_utils.item_width(viz.app.font_size*5):
                changed, idx = imgui.input_int("index", self.params.index+1)
                if changed:
                    self.params.index = int((idx-1) % self.params.num_keyframes)
            imgui.same_line()
            with imgui_utils.item_width(viz.app.font_size*5):
                changed, self.alpha = imgui.slider_float("alpha", self.alpha, 0, 1)

            _, self.perfect_loop = imgui.checkbox("Perfect Loop", self.perfect_loop)

            if self.params.anim:
                self.update_alpha()
                viz.args.alpha = self.alpha
                viz.args.looping_modes = self.modes
                viz.args.looping_seeds = self.seeds
                viz.args.looping_keyframes = self.keyframes
                viz.args.looping_index = self.params.index
                viz.args.looping_projections = self.project
        self.osc_menu()

    def evaluate(self, seed_idx):
        if not self.viz.vm is None:
            latent = self.seeds[seed_idx]
            w0_seeds = []
            for ofs_x, ofs_y in [[0, 0], [1, 0], [0, 1], [1, 1]]:
                seed_x = np.floor(latent.x) + ofs_x
                seed_y = np.floor(latent.y) + ofs_y
                seed = (int(seed_x) + int(seed_y) * self.step_y) & ((1 << 32) - 1)
                weight = (1 - abs(latent.x - seed_x)) * (1 - abs(latent.y - seed_y))
                if weight > 0:
                    w0_seeds.append([seed, weight])
            all_seeds = [seed for seed, _weight in w0_seeds]  # + [mappingmix_seed]
            all_seeds = list(set(all_seeds))
            all_zs = np.zeros([len(all_seeds), self.viz.vm.mapping_dim], dtype=np.float32)
            for idx, seed in enumerate(all_seeds):
                rnd = np.random.RandomState(seed)
                all_zs[idx] = rnd.randn(self.viz.vm.w_dim)

            if not self.project[seed_idx]:
                all_ws = self.to_device(torch.from_numpy(all_zs))
                all_ws = dict(zip(all_seeds, all_ws))
                w = torch.stack([all_ws[seed] * weight for seed, weight in w0_seeds]).sum(dim=0, keepdim=True)
            # Run mapping network.
            else:
                w_avg = self.to_device(self.viz.vm.w_avg)
                all_ws = self.to_device(torch.from_numpy(all_zs))
                all_ws = self.viz.vm.mapping(all_ws) - w_avg
                all_ws = dict(zip(all_seeds, all_ws))
                w = torch.stack([all_ws[seed] * weight for seed, weight in w0_seeds]).sum(dim=0, keepdim=True)
                w += w_avg
            return w

    def snap(self):
        return self.viz.result.get("snap", None)











