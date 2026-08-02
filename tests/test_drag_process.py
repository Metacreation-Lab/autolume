import queue

import pytest
import torch

from drag.engine import DragEngine
from drag.process import drag_session, run_drag


@pytest.fixture(scope='module')
def tiny_g():
    from architectures import custom_stylegan2
    torch.manual_seed(0)
    G = custom_stylegan2.Generator(
        z_dim=64, w_dim=64, c_dim=0, img_resolution=64, img_channels=3,
        mapping_kwargs={'num_layers': 2},
        synthesis_kwargs={'channel_base': 1024, 'channel_max': 64, 'num_fp16_res': 0})
    return G.eval().requires_grad_(False)


def _engine(tiny_g):
    with torch.no_grad():
        w0 = tiny_g.mapping(torch.randn(1, tiny_g.z_dim), None)
    return DragEngine(tiny_g, w0, device='cpu')


def test_session_converged_replies(tiny_g):
    engine = _engine(tiny_g)
    try:
        cmd, reply = queue.Queue(), queue.Queue()
        cfg = {'points': [[32, 32]], 'targets': [[32, 32]], 'mask': None}
        assert drag_session(engine, cfg, cmd, reply) == 'converged'
        msg = reply.get_nowait()
        assert msg['converged'] and msg['step'] == 1
        assert msg['w'].shape == (1, tiny_g.num_ws, tiny_g.w_dim)
        assert msg['w'].dtype.name == 'float32'
    finally:
        engine.close()


def test_session_stop_before_step(tiny_g):
    engine = _engine(tiny_g)
    try:
        cmd, reply = queue.Queue(), queue.Queue()
        cmd.put({'cmd': 'stop'})
        cfg = {'points': [[32, 32]], 'targets': [[44, 32]], 'mask': None}
        assert drag_session(engine, cfg, cmd, reply) == 'stop'
        assert reply.empty()
    finally:
        engine.close()


def test_run_drag_shutdown():
    cmd, reply = queue.Queue(), queue.Queue()
    cmd.put({'cmd': 'shutdown'})
    run_drag(cmd, reply)   # must return, not hang
    assert reply.empty()


def test_run_drag_start_before_load():
    cmd, reply = queue.Queue(), queue.Queue()
    cmd.put({'cmd': 'start', 'w0': None, 'points': [], 'targets': [], 'mask': None})
    cmd.put({'cmd': 'shutdown'})
    assert run_drag(cmd, reply) is None   # keeps serving, exits on shutdown
    assert reply.get_nowait() == {'error': 'start before load'}
    assert reply.empty()
