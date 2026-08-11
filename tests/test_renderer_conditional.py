import torch

import dnnlib
from architectures.networks_stylegan2 import MappingNetwork
from widgets.renderer import Renderer


def make_renderer():
    r = Renderer.__new__(Renderer)
    r._device = torch.device("cpu")
    return r


def make_g(c_dim):
    net = MappingNetwork(z_dim=16, c_dim=c_dim, w_dim=16, num_ws=4,
                         num_layers=1)
    return dnnlib.EasyDict(mapping=net, c_dim=c_dim, num_ws=4)


def test_process_vec_projects_with_conditional_model():
    out = make_renderer().process_vec(make_g(c_dim=3), torch.randn(16),
                                      True, 1, None)
    assert out.shape == (1, 4, 16)


def test_process_vec_projects_with_unconditional_model():
    out = make_renderer().process_vec(make_g(c_dim=0), torch.randn(16),
                                      True, 1, None)
    assert out.shape == (1, 4, 16)


def test_process_vec_without_project():
    out = make_renderer().process_vec(make_g(c_dim=3), torch.randn(16),
                                      False, 1, None)
    assert out.shape == (1, 4, 16)
