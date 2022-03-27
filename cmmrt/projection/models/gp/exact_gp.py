import gpytorch


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

