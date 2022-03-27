from copy import deepcopy
from typing import Union

import torch

from cmmrt.projection.models.preprocessor.rt_transformer import RTTransformer
from cmmrt.projection.models.projector.Projector import Projector
from cmmrt.projection.models.projector.gp_projector import GPProjector
from cmmrt.utils.train.torchutils import to_torch


class RTProjectorPipeline:
    def __init__(self, projector: Projector, x_scaler=Union[None, RTTransformer],
                 y_scaler=Union[None, RTTransformer], finetuning_steps=300, lr=0.01):
        if not isinstance(projector, GPProjector):
            raise NotImplementedError('Only GPProjector is supported at the moment')
        self.projector = projector
        self.projector_weights = deepcopy(projector.state_dict())
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.finetuning_steps = finetuning_steps
        self.lr = lr

    def _scale(self, x, y=None):
        if self.x_scaler is not None:
            x = self.x_scaler.transform(x.reshape(-1, 1))
            x = to_torch(x, next(self.projector.parameters()).device)
        if y is not None and self.y_scaler is not None:
            y = self.y_scaler.transform(y.reshape(-1, 1)).flatten()
            y = to_torch(y, next(self.projector.parameters()).device)
        if y is None:
            return x
        else:
            return x, y

    def _inverse_ci(self, mean, var, z=1.96):
        if self.y_scaler is not None:
            mean, median, lb, ub = self.y_scaler.inverse_ci(mean, var, z=1.96)
            return mean, lb, ub
        else:
            return mean, mean - z * torch.sqrt(var), mean + z * torch.sqrt(var)

    def fit(self, x, y):
        self.projector.prepare_metatesting()
        xf, yf = self._scale(x, y)
        xf = to_torch(xf, next(self.projector.parameters()).device)
        yf = to_torch(yf, next(self.projector.parameters()).device)
        self.projector.load_state_dict(self.projector_weights)
        optimizer = torch.optim.Adam(self.projector.parameters(), lr=self.lr)
        self.projector.train()
        for _ in range(self.finetuning_steps):
            self.projector.train_step(xf, yf, optimizer)
        return self

    def predict(self, x, z=1.96):
        x = to_torch(x, next(self.projector.parameters()).device)
        pred_dist = self._predictive_distribution(x)
        mean, lb, ub = self._inverse_ci(pred_dist.mean.detach().cpu().numpy(),
                                        pred_dist.variance.detach().cpu().numpy(),
                                        z=z)
        return mean, lb, ub

    def _predictive_distribution(self, x):
        self.projector.eval()
        x = self._scale(x)
        with torch.no_grad():
            pred_dist = self.projector(x)
            pred_dist = self.projector.gp.likelihood(pred_dist)
        return pred_dist

    def z_score(self, x, y):
        pred_dist = self._predictive_distribution(x)
        # z_scoring takes places in log space if y_scaler is available
        if self.y_scaler is not None:
            y = self.y_scaler.transform(y.reshape(-1, 1)).flatten()
        y = to_torch(y, next(self.projector.parameters()).device)
        z_score = torch.abs(y - pred_dist.mean) / torch.sqrt(pred_dist.variance)
        return z_score

    def fit_predict(self, x, y):
        return self.fit(x, y).predict(x)

    def load_projector_state_dict(self, state_dict):
        self.projector.load_state_dict(state_dict)
        self.projector_weights = deepcopy(state_dict)
        print("Projector state dict loaded")
