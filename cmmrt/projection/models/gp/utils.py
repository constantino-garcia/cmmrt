import gpytorch
from gpytorch.kernels import ScaleKernel, RBFKernel, LinearKernel, PolynomialKernel, MaternKernel, \
    PiecewisePolynomialKernel, RQKernel

from cmmrt.projection.models.gp.mean import MLPMean


def create_gp_kernel(kernel, dim):
    if kernel == 'rbf':
        kernel = ScaleKernel(RBFKernel(ard_num_dims=dim))
    elif kernel == 'matern_15':
        kernel = ScaleKernel(MaternKernel(ard_num_dims=dim, nu=1.5))
    elif kernel == 'matern_25':
        kernel = ScaleKernel(MaternKernel(ard_num_dims=dim, nu=2.5))
    elif kernel == 'piecewise':
        kernel = ScaleKernel(PiecewisePolynomialKernel(q=2))
    elif kernel == 'rq':
        kernel = ScaleKernel(RQKernel())
    elif kernel == "poly":
        kernel = PolynomialKernel(power=4)
    elif kernel == 'rbf+linear':
        kernel = ScaleKernel(RBFKernel(ard_num_dims=dim)) + LinearKernel(num_dimensions=dim)
    elif kernel == 'rbf+rbf':
        k1 = ScaleKernel(RBFKernel(ard_num_dims=dim))
        k2 = ScaleKernel(RBFKernel(ard_num_dims=dim))
        k2.base_kernel.lengthscale = 0.1
        kernel = k1 + k2
    elif kernel == "rbf*linear":
        kernel = ScaleKernel(RBFKernel(ard_num_dims=dim)) * LinearKernel(num_dimensions=dim)
    elif kernel == 'rbf+poly':
        kernel = ScaleKernel(RBFKernel(ard_num_dims=dim)) + PolynomialKernel(power=4)
    else:
        raise ValueError(f'Invalid kernel {kernel}. Should be one of rbf, rbf+linear, rbf*linear, rbf+poly, poly')
    return kernel


def create_gp_mean(mean, dim):
    if mean == 'zero':
        mean = gpytorch.means.ZeroMean()
    elif mean == 'constant':
        mean = gpytorch.means.ConstantMean()
    elif mean == 'linear':
        mean = gpytorch.means.LinearMean(input_size=dim)
    elif mean == 'mlp':
        mean = MLPMean(dim)
    else:
        raise ValueError('Invalid mean')
    return mean
