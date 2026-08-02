import multiprocessing as mp

import imgui
import numpy as np
import torch

from drag.coords import image_to_screen, screen_to_image
from drag.process import run_drag
from utils.app_logging import LoggedProcess
from utils.gui_utils import gl_utils, imgui_utils


class DragWidget:
    """DragGAN point-based image manipulation. UI modeled on the official
    DragGAN drag widget. Optimization runs in a worker process (drag/process.py);
    this widget streams the evolving W+ into viz.args each frame and commits
    the final latent into the latent widget's vector mode when the drag ends.

    Known limitation: point mapping assumes the rendered image is the
    generator's nominal output size (res.g_dims) times a uniform display zoom.
    With per layer ratio transforms active the mapping is wrong.
    """

    def __init__(self, viz):
        self.viz = viz
        self.armed = False
        self.dragging = False
        self.points = []            # [[y, x], ...] generator pixel coords
        self.targets = []
        self.pending = None         # point being placed while mouse held
        self.expect_target = False
        self.lambda_mask = 20.0
        self.lr = 0.002
        self.step_count = 0
        self.mask = None            # torch [gh, gw] float, 1 = hold fixed
        self.mask_mode = 'point'    # 'point' | 'fixed' | 'flexible'
        self.brush_radius = 50
        self.show_mask = False
        self.g_dims = None          # (gh, gw)
        self._w = None              # np [1, L, 512], latest from worker
        self._w0 = None
        self._d0 = None             # np [512], adjuster direction baked into _w0
        self._linger = False        # keep overriding args one frame post-commit
        self._error = ''
        self._ready = False
        self._process = None
        self._cmd = None
        self._reply = None
        self._proc_pkl = None
        self._failed_pkl = None     # pkl whose worker died, do not auto respawn
        self._mask_tex = None

    # ---- worker lifecycle ----

    def _spawn(self, pkl):
        self._kill()
        self._error = ''
        self._failed_pkl = None
        self._cmd = mp.Queue()
        self._reply = mp.Queue()
        self._process = LoggedProcess(target=run_drag, args=(self._cmd, self._reply),
                                      daemon=True, name='drag')
        self._process.start()
        self._cmd.put({'cmd': 'load', 'pkl': str(pkl),
                       'device': self.viz.args.get('device')})
        self._proc_pkl = pkl

    def _kill(self):
        process, cmd, reply = self._process, self._cmd, self._reply
        self._process = None
        self._cmd = None
        self._reply = None
        self._proc_pkl = None
        self._ready = False
        self.dragging = False
        if process is None:
            return
        try:
            if cmd is not None:
                cmd.put({'cmd': 'shutdown'})
            process.join(timeout=0.2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=0.5)
        except Exception as e:
            self.viz.print_error(e)
        # Drop anything still buffered so the parent never blocks on a feeder
        # thread for a worker that is gone.
        for queue in (cmd, reply):
            if queue is not None:
                queue.cancel_join_thread()

    def close(self):
        self._kill()

    def set_armed(self, armed):
        if armed == self.armed:
            return
        self.armed = armed
        if armed:
            pkl = self.viz.args.get('pkl')
            if pkl:
                self._spawn(pkl)
        else:
            self.stop_drag(commit=True)
            self._kill()
            self.reset_points()
            self._error = ''
            self._failed_pkl = None

    # ---- session ----

    def reset_points(self):
        self.points = []
        self.targets = []
        self.pending = None
        self.expect_target = False
        self.step_count = 0

    def start_drag(self):
        # Poll first so replies left over from the previous session are dropped
        # (step replies are ignored while not dragging) instead of landing in
        # the new one.
        self._poll()
        result = self.viz.result
        n_pairs = min(len(self.points), len(self.targets))
        if not self._ready or self._cmd is None or result is None or 'w' not in result or n_pairs == 0:
            return
        self.points = self.points[:n_pairs]
        self.targets = self.targets[:n_pairs]
        self._w0 = np.array(result.w, dtype=np.float32, copy=True)
        self._w = np.array(result.w, dtype=np.float32, copy=True)
        self._d0 = self._current_direction()
        mask = self.mask.numpy() if self.mask is not None else None
        self._cmd.put({'cmd': 'start', 'w0': self._w0,
                       'points': [list(p) for p in self.points],
                       'targets': [list(t) for t in self.targets],
                       'mask': mask, 'lambda_mask': float(self.lambda_mask),
                       'lr': float(self.lr)})
        self.step_count = 0
        self.dragging = True

    def _current_direction(self):
        direction = self.viz.args.get('direction')
        if direction is None:
            return None
        if torch.is_tensor(direction):
            direction = direction.detach().cpu().numpy()
        return np.array(direction, dtype=np.float32)

    def stop_drag(self, commit=True):
        if not self.dragging:
            return
        self.dragging = False
        if self._cmd is not None:
            self._cmd.put({'cmd': 'stop'})
        if commit and self._w is not None:
            latent = self.viz.latent_widget.latent
            latent.mode = False     # switch to vector mode
            vec = torch.from_numpy(self._w[0].copy())   # [num_ws, 512]
            if self._d0 is not None:
                # res.w was captured after the renderer added the adjuster
                # direction, and the renderer adds it again to whatever the
                # latent widget serves, so shed the copy already baked in.
                vec = vec - torch.from_numpy(self._d0)
            latent.vec = vec
            self._linger = True

    def _poll(self):
        while self._reply is not None and not self._reply.empty():
            msg = self._reply.get()
            error = msg.get('error')
            if error == 'start before load':
                # The only error the worker survives, so keep it running.
                self.dragging = False
                self.viz.print_error('Drag started before the model finished loading')
            elif error:
                self._error = error.splitlines()[-1]
                self.viz.print_error(error)
                self._failed_pkl = self._proc_pkl
                self._kill()
            elif msg.get('ready'):
                self._ready = True
            elif 'step' in msg and self.dragging:
                self.step_count = msg['step']
                self._w = msg['w']
                self.points = [list(p) for p in msg['points']]
                if msg['converged']:
                    self.stop_drag(commit=True)

    # ---- image interaction (called by the visualizer) ----

    def wants_input(self):
        return self.armed and not self._error

    def on_image_click(self, clicked, down, mouse_x, mouse_y, image_area):
        if self.g_dims is None or image_area is None:
            return
        if image_area[2] <= 0 or image_area[3] <= 0:
            return
        gh, gw = self.g_dims
        py, px, inside = screen_to_image(mouse_x, mouse_y, image_area, (gw, gh))
        if not inside:
            return
        if self.mask_mode in ('fixed', 'flexible'):
            if down:
                self._paint(py, px)
            return
        if clicked:
            self.pending = [int(round(py)), int(round(px))]
        elif down and self.pending is not None:
            self.pending = [int(round(py)), int(round(px))]
        elif self.pending is not None:
            if self.dragging:
                self.stop_drag(commit=True)
            if self.expect_target:
                self.targets.append(self.pending)
            else:
                self.points.append(self.pending)
            self.expect_target = not self.expect_target
            self.pending = None

    def _paint(self, py, px):
        gh, gw = self.g_dims
        if self.mask is None or tuple(self.mask.shape) != (gh, gw):
            self.mask = torch.ones(gh, gw)
        yy, xx = torch.meshgrid(torch.arange(gh, dtype=torch.float32),
                                torch.arange(gw, dtype=torch.float32), indexing='ij')
        circle = (yy - py) ** 2 + (xx - px) ** 2 < float(self.brush_radius) ** 2
        self.mask[circle] = 0.0 if self.mask_mode == 'flexible' else 1.0

    # ---- overlay (called by the visualizer after the image blit) ----

    def draw_overlay(self, image_area):
        if not self.armed or self.g_dims is None or image_area is None:
            return
        if image_area[2] <= 0 or image_area[3] <= 0:
            return
        gh, gw = self.g_dims
        size = (gw, gh)
        scale = image_area[2] / gw
        radius = max(4.0, 5.0 * scale)
        points = [list(p) for p in self.points]
        targets = [list(t) for t in self.targets]
        if self.pending is not None:
            (targets if self.expect_target else points).append(self.pending)
        if self.show_mask and self.mask is not None:
            overlay = ((1 - self.mask) * 255).to(torch.uint8).unsqueeze(-1).numpy()
            if self._mask_tex is None or not self._mask_tex.is_compatible(image=overlay):
                self._mask_tex = gl_utils.Texture(image=overlay, bilinear=False, mipmap=False)
            else:
                self._mask_tex.update(overlay)
            center = np.array([image_area[0] + image_area[2] / 2,
                               image_area[1] + image_area[3] / 2])
            self._mask_tex.draw(pos=center, zoom=image_area[2] / gw, align=0.5,
                                rint=True, alpha=0.15)
        for point, target in zip(points, targets):
            sx, sy = image_to_screen(point[0], point[1], image_area, size)
            tx, ty = image_to_screen(target[0], target[1], image_area, size)
            gl_utils.draw_arrow(sx, sy, tx, ty, width=max(2.0, 2.0 * scale),
                                head=2.0 * radius, color=[1, 1, 1], alpha=0.8)
        for point in points:
            sx, sy = image_to_screen(point[0], point[1], image_area, size)
            gl_utils.draw_circle(center=np.array([sx, sy]), radius=radius, color=[1, 0, 0])
        for target in targets:
            sx, sy = image_to_screen(target[0], target[1], image_area, size)
            gl_utils.draw_circle(center=np.array([sx, sy]), radius=radius, color=[0, 0.4, 1])

    # ---- panel ----

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        self._poll()
        result = getattr(viz, 'result', None)
        if result is not None and 'g_dims' in result:
            g_dims = (int(result.g_dims[0]), int(result.g_dims[1]))
            if self.g_dims != g_dims:
                self.g_dims = g_dims
                self.reset_points()
                self.mask = None
        has_model = viz.args.get('pkl') is not None
        if show:
            with imgui_utils.grayed_out(not (has_model or self.armed)):
                _clicked, armed = imgui.checkbox('Enable drag##drag', self.armed)
            if _clicked and (has_model or not armed):
                self.set_armed(armed)
            imgui.same_line()
            if self.dragging:
                imgui.text('Dragging')
            elif self._error:
                imgui.text('Failed')
            elif self.armed and not self._ready:
                imgui.text('Loading model')
            elif self.armed:
                imgui.text('Ready')
            else:
                imgui.text('Off')
            enabled = self.armed and self._ready
            with imgui_utils.grayed_out(not enabled):
                imgui.text('Click the image to place a handle point. Click again for its target.')
                n_pairs = min(len(self.points), len(self.targets))
                if imgui_utils.button('Start##drag' if not self.dragging else 'Stop##drag',
                                      width=viz.app.button_w,
                                      enabled=enabled and (self.dragging or n_pairs > 0)):
                    if self.dragging:
                        self.stop_drag(commit=True)
                    else:
                        self.start_drag()
                imgui.same_line()
                if imgui_utils.button('Revert##drag', width=viz.app.button_w,
                                      enabled=enabled and self.dragging):
                    self.stop_drag(commit=False)
                imgui.same_line()
                if imgui_utils.button('Reset points##drag', width=viz.app.button_w, enabled=enabled):
                    self.reset_points()
                imgui.same_line()
                imgui.text(f'Steps: {self.step_count}')
                with imgui_utils.item_width(viz.app.font_size * 6):
                    _c, self.lr = imgui_utils.input_float('Step size##drag', self.lr, format='%.4f')
                imgui.text('Mask')
                imgui.same_line()
                if imgui_utils.button('Points##dragmode', width=viz.app.button_w, enabled=enabled):
                    self.mask_mode = 'point'
                    self.pending = None
                imgui.same_line()
                if imgui_utils.button('Fixed area##drag', width=viz.app.button_w, enabled=enabled):
                    self.mask_mode = 'fixed'
                    self.show_mask = True
                    self.pending = None
                imgui.same_line()
                if imgui_utils.button('Flexible area##drag', width=viz.app.button_w, enabled=enabled):
                    self.mask_mode = 'flexible'
                    self.show_mask = True
                    self.pending = None
                imgui.same_line()
                if imgui_utils.button('Reset mask##drag', width=viz.app.button_w, enabled=enabled):
                    self.mask = None
                _c, self.show_mask = imgui.checkbox('Show mask##drag', self.show_mask)
                imgui.same_line()
                with imgui_utils.item_width(viz.app.font_size * 5):
                    _c, self.brush_radius = imgui.input_int('Brush##drag', self.brush_radius)
                imgui.same_line()
                with imgui_utils.item_width(viz.app.font_size * 5):
                    _c, self.lambda_mask = imgui_utils.input_float('Mask weight##drag', self.lambda_mask)
                if self._error:
                    imgui.text('Drag failed. See the log for details.')

        # Keep the worker in sync with the loaded model. A worker that died on a
        # model is not respawned for that same model, only a user re-arm retries.
        if self.armed:
            pkl = viz.args.get('pkl')
            if pkl and pkl != self._proc_pkl and pkl != self._failed_pkl:
                self.stop_drag(commit=False)
                self.reset_points()
                self._spawn(pkl)

        # Override the latent while a drag session runs (and one extra frame
        # after commit, until the latent widget serves the committed vector).
        if (self.dragging or self._linger) and self._w is not None:
            viz.args.mode = 'vec'
            viz.args.vec = torch.from_numpy(self._w.copy())   # [1, L, 512] passthrough
            viz.args.project = False
            viz.args.direction = torch.zeros(512)
            if not self.dragging:
                self._linger = False
