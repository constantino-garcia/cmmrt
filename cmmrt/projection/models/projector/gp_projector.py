import gpytorch
import gpytorch.likelihoods.noise_models
import torch
import torch.nn.functional as F
from torch import nn

from cmmrt.projection.models.gp.exact_gp import ExactGPModel
from cmmrt.projection.models.gp.utils import create_gp_kernel, create_gp_mean
from cmmrt.projection.models.projector.Projector import Projector
from cmmrt.utils.train.torchutils import get_default_device


class GPProjector(gpytorch.Module, Projector):
    """Combines a feature extractor and a GP to create a GP with a deep kernel"""

    def __init__(self, mean_name, kernel_name, device=get_default_device()):
        """
        """
        super().__init__()
        self.mean_name = mean_name
        self.kernel_name = kernel_name
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.gp = ExactGPModel(
            mean=create_gp_mean(mean_name, dim=1),
            kernel=create_gp_kernel(kernel_name, dim=1),
            likelihood=self.likelihood,
            train_x=None,
            train_y=None
        )
        self.device = device
        if self.device == 'cuda':
            self.gp.cuda()
            self.likelihood.cuda()
        self.mll = gpytorch.mlls.LeaveOneOutPseudoLikelihood(
            likelihood=self.gp.likelihood,
            model=self.gp
        )

    def forward(self, x):
        return self.gp(x)

    def set_train_data(self, x, y, strict=False):
        self.gp.set_train_data(x, y, strict=strict)

    def train_step(self, x, y, optimizer, **kwargs):
        if self.device == 'cuda':
            x = x.cuda()
            y = y.cuda()
        self.gp.set_train_data(x, y, strict=False)
        self.gp.train()
        optimizer.zero_grad()
        output = self.gp(x)
        loss = -self.mll(output, y)
        loss.backward()
        optimizer.step()
        return loss

    def metaparams(self):
        parameters = self.named_parameters()
        return {k: v for k, v in parameters}

    def update_metaparams(self, new_metaparams):
        state_dict = self.state_dict()
        state_dict.update(new_metaparams)
        self.load_state_dict(state_dict)

    def prepare_metatraining(self):
        pass

    def prepare_metatesting(self):
        for param in self.parameters():
            param.requires_grad = False
        self.likelihood.raw_noise.requires_grad = True


