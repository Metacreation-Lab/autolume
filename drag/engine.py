"""Point-based latent optimization from DragGAN (Pan et al., SIGGRAPH 2023).

Motion supervision and point tracking adapted from the official
implementation (https://github.com/XingangPan/DragGAN, viz/renderer.py,
CC-BY-NC). See the licensing note in the DragGAN plan before relicensing.
"""
import torch
import torch.nn.functional as F


class _CaptureDone(Exception):
    def __init__(self, feat):
        super().__init__()
        self.feat = feat


def track_points(feat, feat_refs, points, r2_px):
    """Relocate each handle point to the best feature match of its original
    feature vector within a square search window of radius r2_px.

    feat: [1, C, H, W] float. feat_refs: list of [1, C] tensors captured at
    the original point locations. points: [[y, x], ...] pixel coords.
    Returns the updated points (ints).
    """
    _, _, h, w = feat.shape
    out = []
    for j, point in enumerate(points):
        py, px = round(point[0]), round(point[1])
        up = max(py - r2_px, 0)
        down = min(py + r2_px + 1, h)
        left = max(px - r2_px, 0)
        right = min(px + r2_px + 1, w)
        patch = feat[:, :, up:down, left:right]
        dist = torch.linalg.norm(patch - feat_refs[j].reshape(1, -1, 1, 1), dim=1)
        idx = torch.argmin(dist.view(-1)).item()
        width = right - left
        out.append([idx // width + up, idx % width + left])
    return out


def motion_loss(feat, points, targets, r1_px, stop_px):
    """Motion supervision: for each handle point, pull the feature patch of
    radius r1_px around it one unit step toward its target by matching the
    detached current patch against the patch sampled one step ahead.

    Returns (loss, converged). converged is True when every point lies
    within stop_px of its target. Points closer than 1 px contribute no loss.
    """
    _, _, h, w = feat.shape
    device = feat.device
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, dtype=torch.float32, device=device),
        torch.arange(w, dtype=torch.float32, device=device),
        indexing='ij')
    loss = feat.sum() * 0.0
    converged = True
    for point, target in zip(points, targets):
        step = torch.tensor([target[0] - point[0], target[1] - point[1]],
                            dtype=torch.float32, device=device)
        dist_to_target = torch.linalg.norm(step)
        if dist_to_target > stop_px:
            converged = False
        if dist_to_target <= 1:
            continue
        step = step / (dist_to_target + 1e-7)
        around = ((grid_y - point[0]) ** 2 + (grid_x - point[1]) ** 2).sqrt() < r1_px
        qy, qx = torch.where(around)
        norm_x = (qx.float() + step[1]) / (w - 1) * 2 - 1
        norm_y = (qy.float() + step[0]) / (h - 1) * 2 - 1
        grid = torch.stack([norm_x, norm_y], dim=-1)[None, None]
        shifted = F.grid_sample(feat, grid, align_corners=True).squeeze(2)
        loss = loss + F.l1_loss(shifted, feat[:, :, qy, qx].detach())
    return loss, converged


class DragEngine:
    """Owns a frozen generator and an optimizable W+ latent. step() runs one
    DragGAN iteration: forward to the feature block, track points, motion
    supervision loss, one Adam step on the first trainable_ws layers.

    The forward pass is short-circuited at the feature block with a hook
    that raises, so blocks above feature_res are never computed.
    """

    def __init__(self, G, w0, device, lr=2e-3, trainable_ws=6, feature_res=128):
        self.G = G
        self.device = torch.device(device)
        w0 = torch.as_tensor(w0, dtype=torch.float32, device=self.device)
        assert w0.ndim == 3 and w0.shape[0] == 1 and w0.shape[1] == G.num_ws
        self.w0 = w0.detach().clone()
        self.trainable_ws = min(trainable_ws, G.num_ws)
        self.w = self.w0.clone().requires_grad_(True)
        self.opt = torch.optim.Adam([self.w], lr=lr)
        candidates = [r for r in G.synthesis.block_resolutions if r <= feature_res]
        self.feature_res = candidates[-1] if candidates else G.synthesis.block_resolutions[0]
        block = getattr(G.synthesis, f'b{self.feature_res}')
        self._capturing = False
        self._hook = block.register_forward_hook(self._on_block)
        self.feat_refs = None
        self.feat0 = None

    def _on_block(self, module, inputs, outputs):
        # Inert unless this engine is mid-capture, so the hook never disturbs
        # another engine's or a third party's forward through the same G.
        if not self._capturing:
            return
        feat = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        raise _CaptureDone(feat)

    def close(self):
        self._hook.remove()

    def current_w(self):
        with torch.no_grad():
            return torch.cat([self.w[:, :self.trainable_ws],
                              self.w0[:, self.trainable_ws:]], dim=1).detach().clone()

    def _features(self):
        ws = torch.cat([self.w[:, :self.trainable_ws],
                        self.w0[:, self.trainable_ws:]], dim=1)
        self._capturing = True
        try:
            self.G.synthesis(ws, noise_mode='const', force_fp32=True)
        except _CaptureDone as e:
            return e.feat.float()
        finally:
            self._capturing = False
        raise RuntimeError(f'feature block b{self.feature_res} did not run')

    def step(self, points, targets, mask=None, lambda_mask=20.0, r1=3, r2=12):
        """One optimization iteration. points/targets in generator pixel
        coords [[y, x], ...]. mask [gh, gw], 1 = hold fixed. The mask only
        applies when it contains both 0s and 1s, so an untouched all-ones
        mask is a no-op (matches the official implementation).
        Returns (points, converged); w is not updated once converged.
        """
        h, w = int(self.G.output_shape[2]), int(self.G.output_shape[3])
        scale = min(h, w)
        feat = self._features()
        feat_resize = F.interpolate(feat, [h, w], mode='bilinear')
        if self.feat_refs is None:
            self.feat0 = feat_resize.detach()
            self.feat_refs = [self.feat0[:, :, round(p[0]), round(p[1])] for p in points]

        with torch.no_grad():
            points = track_points(feat_resize, self.feat_refs, points,
                                  r2_px=max(round(r2 / 512 * scale), 1))

        loss, converged = motion_loss(feat_resize, points, targets,
                                      r1_px=max(round(r1 / 512 * scale), 1),
                                      stop_px=max(2 / 512 * scale, 2))
        if mask is not None and mask.min() == 0 and mask.max() == 1:
            m = mask.to(device=feat_resize.device, dtype=feat_resize.dtype)[None, None]
            loss = loss + lambda_mask * F.l1_loss(feat_resize * m, self.feat0 * m)
        if not converged:
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        return points, converged
