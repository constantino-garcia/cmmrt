import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from torch.optim.swa_utils import AveragedModel
from torch.optim.swa_utils import SWALR
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from cmmrt.rt.models.base.PipelineWrapper import RTRegressor


class _DnnModel(nn.Module):
    def __init__(self, n_features, hidden_1=1512, hidden_2=128, dropout_1=0.5, dropout_2=0.1, activation='gelu'):
        super().__init__()
        self.l1 = nn.Linear(n_features, hidden_1)
        nn.init.zeros_(self.l1.bias)
        self.d1 = nn.Dropout(dropout_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        nn.init.zeros_(self.l2.bias)
        self.d2 = nn.Dropout(dropout_2)
        self.l_out = nn.Linear(hidden_2, 1)
        nn.init.zeros_(self.l_out.bias)
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'swish':
            self.activation = F.silu

    def forward(self, x):
        x = self.d1(self.activation(self.l1(x)))
        x = self.d2(self.activation(self.l2(x)))
        return self.l_out(x)


class _SkDnn(BaseEstimator, RegressorMixin):
    def __init__(self, hidden_1=1512, hidden_2=128, dropout_1=0.5, dropout_2=0.1, activation='gelu',
                 lr=3e-4, T0=30, annealing_rounds=2, swa_epochs=20, batch_size=64, device='cuda'):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)
        self.n_epochs = self.annealing_rounds * self.T0

    def _init_hidden_model(self, n_features):
        self._model = _DnnModel(n_features).to(self.device)
        min_lr = 0.1 * self.lr
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self._optimizer, T_0=self.T0, T_mult=1, eta_min=min_lr
        )
        self._swa_model = AveragedModel(self._model)
        self._swa_scheduler = SWALR(self._optimizer, swa_lr=min_lr)

    def fit(self, X, y):
        self._init_hidden_model(X.shape[1])
        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).view(-1, 1))
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self._model.train()
        train_iters = len(data_loader)
        for epoch in range(self.n_epochs):
            for i, (xb, yb) in enumerate(data_loader):
                self._batch_step(xb, yb)
                self._scheduler.step(epoch + i / train_iters)

        self._swa_model.train()
        for epoch in range(self.swa_epochs):
            for xb, yb in data_loader:
                self._batch_step(xb, yb)
            self._swa_model.update_parameters(self._model)
            self._swa_scheduler.step()

        return self

    def _batch_step(self, xb, yb):
        self._optimizer.zero_grad()
        pred = self._model(xb.to(self.device))
        loss = F.l1_loss(pred, target=yb.to(self.device))
        loss.backward()
        self._optimizer.step()

    def predict(self, X):
        self._model.eval()
        self._swa_model.eval()
        with torch.no_grad():
            return self._swa_model(torch.from_numpy(X).to(self.device)).cpu().numpy().flatten()

    def __getstate__(self):
        state = super().__getstate__().copy()
        if '_model' in state.keys():
            for key in ['_model', '_optimizer', '_scheduler', '_swa_model', '_swa_scheduler']:
                state[key] = state[key].state_dict()
        return state

    def __setstate__(self, state):
        self.__dict__ = state.copy()
        if '_model' in state.keys():
            self._init_hidden_model(state['_model']['l1.weight'].shape[1])
            for key in ['_model', '_optimizer', '_scheduler', '_swa_model', '_swa_scheduler']:
                torch_model = getattr(self, key)
                torch_model.load_state_dict(state.pop(key))
                setattr(self, key, torch_model)


class SkDnn(RTRegressor):
    def __init__(self, hidden_1=1512, hidden_2=128, dropout_1=0.5, dropout_2=0.1, activation='gelu',
                 lr=3e-4, T0=30, annealing_rounds=2, swa_epochs=20, batch_size=64, device='cuda',
                 use_col_indices='all', binary_col_indices=None, var_p=0, transform_output=False
                 ):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)

    def _init_regressor(self):
        return _SkDnn(**self._rt_regressor_params())
