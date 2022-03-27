import torch

from cmmrt.projection.metatrainers.naive import NaiveMetaTrainer


def create_metatrainer(metatrainer_name, projector, inner_optimizer, inner_epochs, outer_epochs, device):
    if metatrainer_name == 'naive':
        metatrainer = NaiveMetaTrainer(projector, inner_optimizer, inner_epochs=inner_epochs,
                                       outer_epochs=outer_epochs, device=device)
    else:
        raise ValueError(f"Unknown metatrainer {metatrainer_name}")
    return metatrainer
