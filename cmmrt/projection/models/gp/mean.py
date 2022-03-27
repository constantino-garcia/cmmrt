"""Classes for projecting retention times between chromatograhic methods."""

import gpytorch
import torch.nn as nn


class MLPMean(gpytorch.means.Mean):
    """Learnable mean function for GP regression based on a multilayer perceptron."""

    def __init__(self, dim, hidden_dim=128, activation='relu'):
        super(MLPMean, self).__init__()
        if activation == 'relu':
            activation_1, activation_2 = nn.LeakyReLU(), nn.LeakyReLU()
        elif activation == 'elu':
            activation_1, activation_2 = nn.ELU(), nn.ELU()
        elif activation == 'selu':
            activation_1, activation_2 = nn.SELU(), nn.SELU()
        elif activation == 'gelu':
            print('GELU')
            activation_1, activation_2 = nn.GELU(), nn.GELU()
        else:
            raise ValueError('Invalid activation function.')

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            activation_1,
            nn.Linear(hidden_dim, hidden_dim),
            activation_2,
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        m = self.mlp(x)
        return m.squeeze()
