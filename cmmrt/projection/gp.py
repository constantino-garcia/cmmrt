"""Classes for projecting retention times between chromatograhic methods."""
import gpytorch
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPMean(gpytorch.means.Mean):
    """Learnable mean function for GP regression based on a multilayer perceptron."""
    def __init__(self, dim):
        super(MLPMean, self).__init__()
        HD = 128
        self.mlp = nn.Sequential(
            nn.Linear(dim, HD),
            nn.LeakyReLU(),
            nn.Linear(HD, HD),
            nn.LeakyReLU(),
            nn.Linear(HD, 1)
        )

    def forward(self, x):
        m = self.mlp(x)
        return m.squeeze()


class DKLProjector(gpytorch.Module):
    """Combines a feature extractor and a GP to create a GP with a deep kernel"""
    def __init__(self, feature_extractor, gp):
        """
        :param feature_extractor: torch.nn.Module implementing a feature extractor. May be None (no feature extractor is used).
        :param gp: A GP to be used on top of the feature extractor.
        """
        super().__init__()
        self.feature_extractor = feature_extractor
        self.gp = gp

    def forward(self, x):
        if self.training:
            return self.gp(self.gp.train_inputs[0])
        else:
            if self.feature_extractor is not None:
                z = self.feature_extractor(x)
            else:
                z = x
            return self.gp(z)

    def set_train_data(self, x, y, strict=False):
        if self.feature_extractor is not None:
            z = self.feature_extractor(x)
        else:
            z = x
        self.gp.set_train_data(z, y, strict=strict)


class ExactGPModel(gpytorch.models.ExactGP):
    """GP model with exact posterior calculation."""
    def __init__(self, mean, kernel, likelihood, train_x, train_y):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = mean
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class FeatureExtractor(nn.Module):
    """Feature extractor to be used in DKLProjector."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.h1 = nn.Linear(1, hidden_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.h2 = nn.Linear(hidden_dim, hidden_dim // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.out = nn.Linear(hidden_dim, 2)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        z = self.bn1(self.h1(x))
        z = F.leaky_relu(z)
        z = self.bn2(self.h2(z))
        z_agg = torch.mean(z, axis=0)
        z = torch.cat([z, z_agg.unsqueeze(0).repeat(x.shape[0], 1)], axis=1)
        z_out = self.out(z)
        return z_out
