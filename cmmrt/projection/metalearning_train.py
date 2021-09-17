import os
import warnings

import gpytorch
import pandas as pd
import torch
from gpytorch.kernels import SpectralMixtureKernel, RBFKernel, PolynomialKernel, LinearKernel, ScaleKernel
from torch.utils.data import DataLoader

from cmmrt.projection.data import load_xabier_projections, ProjectionsTasks, Detrender
from cmmrt.projection.gp import ExactGPModel, FeatureExtractor, DKLProjector, MLPMean
from cmmrt.utils.generic_utils import handle_saving_dir

warnings.simplefilter("ignore")


def meta_train_gp(dat, scaler, use_feature_extraction=True, mean='zero',
                  kernel='spectral_mixture', num_mixtures=4,
                  p_support_range=(1.0, 1.0), max_epochs=156, device='cuda'):
    tasks = ProjectionsTasks(dat, p_support_range=p_support_range, scaler=scaler)
    tasks_dl = DataLoader(tasks, batch_size=1, shuffle=True)

    dim = 2 if use_feature_extraction else 1
    if mean == 'zero':
        mean = gpytorch.means.ZeroMean()
    elif mean == 'linear':
        mean = gpytorch.means.LinearMean(input_size=dim)
    elif mean == 'mlp':
        mean = MLPMean(dim)
    else:
        raise ValueError('Invalid mean')

    if kernel == 'spectral_mixture':
        kernel = SpectralMixtureKernel(num_mixtures=num_mixtures, ard_num_dims=dim)
        train_x, train_y = zip(*[batch for batch in tasks_dl])
        train_x = torch.cat(train_x, axis=1).squeeze(0)
        train_y = torch.cat(train_y, axis=1).squeeze(0)
        kernel.initialize_from_data(train_x, train_y)
        del train_x
        del train_y
    elif kernel == 'rbf':
        kernel = ScaleKernel(RBFKernel(ard_num_dims=dim))
    elif kernel == 'rbf+linear':
        kernel = ScaleKernel(RBFKernel(ard_num_dims=dim)) + ScaleKernel(LinearKernel(num_dimensions=dim))
    elif kernel == 'rbf+poly':
        kernel = ScaleKernel(RBFKernel(ard_num_dims=dim)) + ScaleKernel(PolynomialKernel(power=3))
    elif kernel == "poly":
        kernel = ScaleKernel(PolynomialKernel(power=4))
    else:
        raise ValueError('Invalid kernel. Should be spectral_mixture or RBF')
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    gp = ExactGPModel(mean, kernel, likelihood, None, None)
    if use_feature_extraction:
        feature_extractor = FeatureExtractor(256)
    else:
        feature_extractor = None
    model = DKLProjector(feature_extractor, gp)

    if device == 'cuda':
        model = model.cuda()
    # mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model.gp)
    mll = gpytorch.mlls.LeaveOneOutPseudoLikelihood(likelihood, model.gp)

    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': 0.001}
    ])
    model.train()

    for epoch in range(1, max_epochs + 1):
        total_loss = 0
        for x, y in tasks_dl:
            x = x.squeeze(0)
            y = y.squeeze(0)

            detrender = Detrender()
            y = detrender.fit_transform(x, y)

            if device == 'cuda':
                x = x.cuda()
                y = y.cuda()

            optimizer.zero_grad()
            model.set_train_data(x, y, strict=False)
            predictions = model(x)
            loss = -mll(predictions, y)
            total_loss += loss
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print('[%d] - Loss: %.3f  noise: %.3f' % (
                epoch, total_loss.item(), model.gp.likelihood.noise.item()
            ))

    model.eval()
    return model, mll


def get_representatives(x, y, test_size, n_quantiles=10):
    x_ndim = x.ndim
    y_ndim = y.ndim
    # First sample() performs stratified sampling, second sample() limits the number
    # of samples to test_size
    # groups = pd.qcut(x.flatten(), n_quantiles)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_quantiles).fit(x.reshape(-1, 1))
    groups = kmeans.labels_

    df = pd.DataFrame({'x': x.flatten(), 'y': y.flatten(), 'group': groups})
    representatives = df.groupby(df.group).sample(1)
    while len(representatives) < test_size:
        n_to_go = test_size - len(representatives)
        not_sampled = df[~df.isin(representatives).all(1)]
        new_samples = not_sampled.groupby(not_sampled.group).sample(1)
        if len(new_samples) > n_to_go:
            new_samples = new_samples.sample(n_to_go)
        representatives = pd.concat([representatives, new_samples], axis=0)

    xr = representatives.x.values
    yr = representatives.y.values
    if x_ndim == 2:
        xr = xr.reshape(-1, 1)
    if y_ndim == 2:
        yr = yr.reshape(-1, 1)
    return xr, yr


def create_parser():
    import argparse
    my_parser = argparse.ArgumentParser(description='meta-train gp on projection tasks')
    my_parser.add_argument('-e', '--epochs', type=int, default=150)
    my_parser.add_argument('-d', '--device', type=str, default='cpu')
    my_parser.add_argument('-m', '--mean', type=str, default='zero')  # mlp_mean or not
    my_parser.add_argument('-f', '--feat', action='store_true')  # feature_extraction or not
    my_parser.add_argument('-k', '--kernel', type=str, default='rbf+linear')
    my_parser.add_argument('-r', '--results', help='file prefix for the figures/results',
                           type=str)
    my_parser.add_argument('-s', '--save_to', type=str, default='.', help='folder where to save the figures/results')
    return my_parser


def get_basename(args):
    filename = (args.mean + '_fe') if args.feat else args.mean
    filename += ('_' + args.kernel)
    filename = os.path.join(args.save_to, filename)
    return filename


def add_timestamp(filename):
    import time
    timestamp = str(int(round(time.time() * 1000)))
    return filename + f"-{timestamp}"


if __name__ == '__main__':
    # Create the parser
    my_parser = create_parser()
    args = my_parser.parse_args()
    handle_saving_dir(args.save_to)
    print(args)

    filename = add_timestamp(get_basename(args))
    train_data, systems, scaler = load_xabier_projections("../../rt_data", remove_non_retained=True)

    num_mixtures = args.num_mixtures if hasattr(args, 'num_mixtures') else 5
    gp, mll = meta_train_gp(
        train_data,
        use_feature_extraction=args.feat,
        mean=args.mean,
        max_epochs=args.epochs,
        kernel=args.kernel,
        num_mixtures=num_mixtures,
        scaler=scaler,
        device=args.device
    )
    save_to = filename + '.pth'
    print(f"Saving model to {save_to}")
    torch.save(gp, save_to)
