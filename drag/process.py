"""Grad-enabled DragGAN worker. Runs in its own multiprocessing process so
the inference-mode render worker stays untouched. Queues carry only plain
Python data and numpy arrays (MPS tensors cannot cross process boundaries).

Commands on cmd_queue:
    {'cmd': 'load', 'pkl': str, 'device': str | None}
    {'cmd': 'start', 'w0': np[1, L, 512], 'points': [[y, x], ...],
     'targets': [[y, x], ...], 'mask': np[gh, gw] | None,
     'lambda_mask': float, 'lr': float}
    {'cmd': 'stop'}        end the running session (ignored between sessions)
    {'cmd': 'shutdown'}    exit the process

Replies on reply_queue:
    {'ready': True} after load
    {'step': int, 'w': np[1, L, 512] float32, 'points': [[y, x], ...],
     'converged': bool} once per optimization step
    {'error': str} full traceback on any exception, then the process returns
    {'error': 'start before load'} when a 'start' arrives before a 'load'. This
        is the one error reply the process survives: it keeps serving commands.
"""
import numpy as np
import torch

import dnnlib
from torch_utils import legacy
from drag.engine import DragEngine
from utils.device_utils import get_device


def drag_session(engine, cfg, cmd_queue, reply_queue):
    """Step until converged or told to stop. Returns the outcome."""
    points = [[int(round(p[0])), int(round(p[1]))] for p in cfg['points']]
    targets = [[int(round(t[0])), int(round(t[1]))] for t in cfg['targets']]
    mask = None
    if cfg.get('mask') is not None:
        mask = torch.as_tensor(np.asarray(cfg['mask']), dtype=torch.float32,
                               device=engine.device)
    step = 0
    while True:
        while not cmd_queue.empty():
            msg = cmd_queue.get()
            if msg['cmd'] in ('stop', 'shutdown'):
                return msg['cmd']
        points, converged = engine.step(points, targets, mask=mask,
                                        lambda_mask=cfg.get('lambda_mask', 20.0))
        step += 1
        reply_queue.put({'step': step,
                         'w': engine.current_w().cpu().numpy(),
                         'points': [list(p) for p in points],
                         'converged': converged})
        if converged:
            return 'converged'


def run_drag(cmd_queue, reply_queue):
    """Multiprocessing entry point. See drag/process.py protocol docstring."""
    G = None
    device = None
    try:
        while True:
            msg = cmd_queue.get()
            cmd = msg.get('cmd')
            if cmd == 'shutdown':
                return
            elif cmd == 'load':
                device = torch.device(msg.get('device') or get_device())
                with dnnlib.util.open_url(msg['pkl'], verbose=False) as f:
                    G = legacy.load_network_pkl(f)['G_ema']
                G = G.eval().requires_grad_(False).to(device)
                reply_queue.put({'ready': True})
            elif cmd == 'start':
                if G is None:
                    reply_queue.put({'error': 'start before load'})
                    continue
                engine = DragEngine(G, msg['w0'], device, lr=msg.get('lr', 2e-3))
                try:
                    outcome = drag_session(engine, msg, cmd_queue, reply_queue)
                finally:
                    engine.close()
                if outcome == 'shutdown':
                    return
            # stray 'stop' between sessions and unknown commands are ignored
    except Exception:
        import traceback
        reply_queue.put({'error': traceback.format_exc()})
