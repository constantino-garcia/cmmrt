import inspect
import os
import shutil
import tempfile

import gpytorch
import torch
import torch.nn.functional as F
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import GridInterpolationVariationalStrategy
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from torch import nn

from cmmrt.rt.models.base.PipelineWrapper import RTRegressor
from cmmrt.utils.train.torchutils import EarlyStopping
from cmmrt.utils.train.torchutils import get_default_device
from cmmrt.utils.train.torchutils import torch_dataloaders


class FeatureExtractor(nn.Module):
    def __init__(self, in_features, out_features,
                 hidden_1=1512, hidden_2=128, dropout=0.5,
                 use_bn_out=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.base = nn.Sequential(
            nn.Linear(in_features, hidden_1, bias=True),
            # nn.BatchNorm1d(sizes[0]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_1, hidden_2, bias=False),
            nn.BatchNorm1d(hidden_2)
        )
        self.use_bn_out = use_bn_out
        if self.use_bn_out:
            self.l_out = nn.Linear(hidden_2, out_features, bias=False)
            self.bn_out = nn.BatchNorm1d(out_features)
        else:
            self.l_out = nn.Linear(hidden_2, out_features, bias=True)

    def forward(self, x):
        x = F.gelu(self.base(x))
        if self.use_bn_out:
            return self.bn_out(self.l_out(x))
        else:
            return self.l_out(x)


class GPRegressionModel(ApproximateGP):
    def __init__(self, kernel, input_dim, grid_bounds=(-5., 5.), grid_size=10):
        variational_distribution = CholeskyVariationalDistribution(num_inducing_points=int(pow(grid_size, input_dim)))
        variational_strategy = GridInterpolationVariationalStrategy(self, grid_size, grid_bounds=[grid_bounds] * input_dim,
                                                                    variational_distribution=variational_distribution)
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel
        self.grid_bounds = grid_bounds

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DKL(nn.Module):
    def __init__(self, feature_extractor, regression_model):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.regression_model = regression_model
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(self.regression_model.grid_bounds[0],
                                                                 self.regression_model.grid_bounds[1])

    def forward(self, x):
        projected_x = self.feature_extractor(x)
        projected_x = self.scale_to_bounds(projected_x)
        return self.regression_model(projected_x)


class _SkDKL(BaseEstimator, RegressorMixin):
    def __init__(self, out_features,
                 kernel='linear', hidden_1=1512, hidden_2=128, dropout=0.5,
                 use_bn_out=False, lr=1e-3, batch_size=512, train_epochs=5000,
                 test_size=0.1, scheduler_patience=10, early_stopping=25, device=get_default_device()):
        assert kernel in ['linear', 'rbf', 'mixture'], 'Invalid kernel'
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)

    def _create_checkpoint(self):
        self._ckpt_folder = tempfile.mkdtemp()
        return os.path.join(self._ckpt_folder, 'early_stopping.pth')

    def _delete_checkpoint(self):
        if os.path.exists(self._ckpt_folder):
            try:
                shutil.rmtree(self._ckpt_folder)
            except OSError as e:
                print(f"Error: {e.filename} - {e.strerror}")

    def _init_models(self, num_data, in_features):
        feature_extractor = FeatureExtractor(
            in_features=in_features, out_features=self.out_features,
            hidden_1=self.hidden_1, hidden_2=self.hidden_2,
            dropout=self.dropout, use_bn_out=self.use_bn_out
        )
        if self.kernel == 'linear':
            kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.LinearKernel()
            )
        elif self.kernel == 'rbf':
            kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=self.out_features)
            )
        elif self.kernel == 'mixture':
            kernel = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4, ard_num_dims=self.out_features)
        else:
            raise ValueError('Invalid kernel for DKL')

        regression_model = GPRegressionModel(
            kernel=kernel, input_dim=self.out_features
        )
        self._dkl = DKL(feature_extractor, regression_model).to(self.device)
        self._likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self._optimizer = torch.optim.Adam([
            {'params': self._dkl.feature_extractor.parameters()},
            {'params': self._dkl.regression_model.parameters()},
            {'params': self._likelihood.parameters()}
        ], lr=self.lr)
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self._optimizer, mode='min',
                                                                     patience=self.scheduler_patience,
                                                                     verbose=False)
        ckpt = self._create_checkpoint()
        self._early_stopping = EarlyStopping(patience=self.early_stopping, path=ckpt, verbose=False)
        self._mll = gpytorch.mlls.VariationalELBO(
            self._likelihood,
            self._dkl.regression_model,
            num_data=num_data
        )

    def fit(self, X, y):
        print(f'SkDKL features-->{X.shape[1]}')
        self._init_models(*X.shape)
        train_loader, test_loader = torch_dataloaders(X, y, self.batch_size, test_size=self.test_size, n_strats=6)
        for epoch in range(self.train_epochs):
            # Train
            self._dkl.feature_extractor.train()
            self._dkl.regression_model.train()
            self._likelihood.train()
            for data, target in train_loader:
                if self.device == 'cuda':
                    data, target = data.cuda(), target.cuda()
                self._optimizer.zero_grad()
                output = self._dkl(data)
                loss = -self._mll(output, target.view(-1, ))
                loss.backward()
                self._optimizer.step()

            # Val
            self._dkl.feature_extractor.eval()
            self._dkl.regression_model.eval()
            self._likelihood.eval()
            targets = []
            predictions = []
            for data, target in test_loader:
                if self.device == 'cuda':
                    data, target = data.cuda(), target.cuda()
                with torch.no_grad():
                    output = self._dkl(data)
                    targets.append(target)
                    predictions.append(output.mean)
            mse = F.mse_loss(torch.cat(predictions).view(-1, 1), torch.cat(targets)).cpu().numpy()
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, mse = {mse}')
            self._scheduler.step(mse)
            self._early_stopping(mse, self._dkl)
            if self._early_stopping.early_stop:
                self._early_stopping.load_checkpoint(self._dkl)
                break
        self._delete_checkpoint()
        self._dkl.feature_extractor.eval()
        self._dkl.regression_model.eval()
        self._likelihood.eval()
        return self

    def predict(self, X):
        self._dkl.eval()
        with torch.no_grad():
            y_preds = self._dkl(torch.tensor(X).to(self.device))
        return y_preds.mean.cpu().numpy().flatten()


class SkDKL(RTRegressor):
    def __init__(self, out_features, kernel='linear', hidden_1=1512, hidden_2=128, dropout=0.5, use_bn_out=False,
                 lr=1e-3, batch_size=512, train_epochs=5000, test_size=0.1, scheduler_patience=10, early_stopping=25,
                 device=get_default_device(), use_col_indices='all', binary_col_indices=None, var_p=0, transform_output=True):
        super().__init__(use_col_indices, binary_col_indices, var_p, transform_output)
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)

    def _init_regressor(self):
        return _SkDKL(**self._rt_regressor_params())
