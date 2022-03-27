import importlib_resources
import torch

from cmmrt.projection.data import load_cmm_projections
from cmmrt.projection.models.projector.gp_projector import GPProjector
from cmmrt.projection.models.projector.sk_rt_projector import RTProjectorPipeline


def _load_projector_pipeline_from(state_dict_path, mean='constant', kernel='poly', device='cpu'):
    projector = GPProjector(mean, kernel, device=device)
    projector.load_state_dict(torch.load(state_dict_path, map_location=torch.device(device)))
    _, _, x_scaler, y_scaler = load_cmm_projections()
    return RTProjectorPipeline(projector, x_scaler, y_scaler)


def load_pretrained_projector_pipeline(direction, device="cpu"):
    if direction == "p2e":
        state_dict_path = importlib_resources.files("cmmrt.data").joinpath("p2e_poly.pt")
    elif direction == "e2p":
        state_dict_path = importlib_resources.files("cmmrt.data").joinpath("e2p_poly.pt")
    else:
        raise ValueError("Direction must be either 'p2e' (predicted 2 experimental)"
                         " or 'e2p' (experimental 2 predicted).")
    return _load_projector_pipeline_from(state_dict_path, 'constant', 'poly', device)
