import pyro.optim
import torch

from cmmrt.projection.models.gp.utils import create_gp_mean, create_gp_kernel
from cmmrt.projection.models.projector.gp_projector import GPProjector


def create_projector_and_optimizer(model_type, mean, kernel,
                                   lr, weight_decay, device):
    if model_type == 'exact':
        projector = GPProjector(mean, kernel, device)
    else:
        raise ValueError("Unknown model type. Currently only exact projectors are supported.")

    if weight_decay > 0:
        unregularized = []
        regularized = []
        for name, p in projector.named_parameters():
            if 'mlp' in name and 'weight' in name:
                print("Adding weight decay to {}".format(name))
                regularized += [p]
            else:
                unregularized += [p]
        inner_optimizer = torch.optim.Adam([
            {'params': unregularized, 'lr': lr},
            {'params': regularized, 'weight_decay': weight_decay, 'lr': lr},

        ])
    else:
        inner_optimizer = torch.optim.Adam(projector.parameters(), lr=lr)
    return projector, inner_optimizer
