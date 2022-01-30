"""Meta-test GP on the PredRet dataset

This script allows the user to meta-test a GP model on the PredRet dataset as done in the paper
<<Domingo-Almenara, Xavier, et al. "The METLIN small molecule dataset for machine learning-based retention
time prediction." Nature communications 10.1 (2019): 1-9>>
(that is, predicted RTs are taken from this paper).

Meta-training is done by using the GP model to create projections between the predicted RTs and the experimental
RTs measured using different chromatographic methods. The result of meta-training is a GP with tuned hyperparameters
that can be interpreted as a sensible prior to be used when creating a projection model from a small amount of data.

The performance of the model is evaluated by
1. selecting a system.
2. Meta-training the GP on the PredRet dataset but excluding the system selected in step 1.
3. Performance is evaluated by:
    3.1. Selecting a small amount of points from the system selected in step 1.
    3.2. Using the meta-trained GP as prior, train it on the points from step 3.1.
    3.3. Evaluate projection error on the points not used in steps 3.1-3.2.

The results from the evaluation are saved in a csv file. PNG figures of the projections are also generated.

This script permits the user to specify command line options. Use
$ python metalearning_test.py --help
to see the options.
"""
import os
import dill as pickle
# import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from cmmrt.projection.data import load_xabier_projections, Detrender
from cmmrt.projection.metalearning_train import (
    get_representatives, meta_train_gp, create_parser, get_basename, add_timestamp
)
from cmmrt.utils.train.torchutils import to_torch
from cmmrt.utils.generic_utils import handle_saving_dir

warnings.simplefilter("ignore")


def plot_projection(proj_data, support_data):
    """Plot the projections learnt using the support data."""
    plt.fill_between(
        proj_data.x.values,
        proj_data.lb.values,
        proj_data.ub.values,
        color="C0",
        alpha=0.2,
    )
    # mean has to be accessed with the name so that is not confused with the method mean()
    plt.plot(proj_data.x.values, proj_data['mean'].values, 'C0', lw=2)
    plt.plot(proj_data.x.values, proj_data.y.values, "kx", mew=2)
    plt.plot(support_data.x.values, support_data.y.values, "rx", mew=2)


def meta_test(gp, mll, system_data, scaler, args, n_annotated_samples, support=None, use_detrender=False):
    """Evaluates the performance of the meta-trained GP on a given system.
    
    :param gp: meta-trained DKLProjector to be used as prior in the projection tasks.
    :param mll: pytorch.mlls.LeaveOneOutPseudoLikelihood representing the loss function used for meta-training.
    :param system_data: pandas dataframe containing the data for a specific system to be used for testing.
    :param scaler: object of class RTTransformer used to scale the RTs.
    :param args: command line arguments.
    :param n_annotated_samples: Number of points to use from the system for creating the projection function. These
    points represent molecules whose identity is known (and hence, whose predicted RTs are known).
    :param support: tuple (x, y) representing a set of points to be used for creating the projection function. If
    specified, test_size is ignored.
    :param use_detrender: boolean indicating if a detrender should be used to center y.
    :return: projections (including confidence intervals), data used for creating the projection function, relative errors,
    loss value.
    """
    x = system_data.rt_pred.values.astype('float32')
    y = system_data.rt_exper.values.astype('float32')
    x = scaler.transform(x.reshape(-1, 1)).flatten()
    y = scaler.transform(y.reshape(-1, 1)).flatten()
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]
    # Note that x and y are swapped on stratified_train_test_split so that stratification is done based on x
    if support is None:
        x_support, y_support = get_representatives(x, y, test_size=n_annotated_samples, n_quantiles=10)
        x_support = x_support.reshape(-1, 1)
    else:
        print('Using Xabier support')
        x_support, y_support = support
        x_support = scaler.transform(x_support)
        y_support = scaler.transform(y_support.reshape(-1, 1)).flatten()

    if use_detrender:
        detrender = Detrender()
        y_support = detrender.fit_transform(x_support, y_support)
        y = detrender.transform(y)

    # print(gp.gp.likelihood.noise_covar.noise.item())
    # print(gp.gp.covar_module)
    gp.set_train_data(to_torch(x_support, args.device), to_torch(y_support, args.device),
                      strict=False)
    gp.eval()
    mll.eval()
    with torch.no_grad():
        preds = gp(to_torch(x.reshape(-1, 1), args.device))
        loss = -mll(preds, to_torch(y, args.device))
    mean, var = preds.mean.cpu().numpy(), preds.variance.cpu().numpy()

    if use_detrender:
        y = detrender.inverse_transform(y)
        y_support = detrender.inverse_transform(y_support)
        mean = detrender.inverse_transform(mean)
        var = detrender.inverse_var_transform(var)

    x = scaler.inverse_transform(x.reshape(-1, 1)).flatten()
    y = scaler.inverse_transform(y.reshape(-1, 1)).flatten()
    x_support = scaler.inverse_transform(x_support.reshape(-1, 1)).flatten()
    y_support = scaler.inverse_transform(y_support.reshape(-1, 1)).flatten()
    new_mean, median, lb, ub = scaler.inverse_ci(mean, var)

    all_data = pd.DataFrame({
        'x': x,
        'y': y,
        'mean': new_mean,
        'med': median,
        'lb': lb,
        'ub': ub
    })
    support_data = pd.DataFrame({
        'x': x_support,
        'y': y_support,
    })

    relative_errors = (
        np.abs(all_data.y.values.flatten() - all_data[['mean']].values.flatten()) / all_data.y.values.flatten()
    )

    return all_data, support_data, relative_errors, loss


def get_test_filename(args, n_annotated_samples, system, timestamp):
    """Get a filename to be used to store testing results based on the command line arguments and the system name."""
    filename = get_basename(args)
    return filename + '-' + str(n_annotated_samples) + '-' + system + '-' + timestamp


if __name__ == '__main__':
    # Create the parser
    my_parser = create_parser()
    my_parser.add_argument('-t', '--n_annotated', type=int, default=0,
                           help='Number of samples to use for creating the projection function.')
    args = my_parser.parse_args()
    handle_saving_dir(args.save_to)
    print(args)

    if args.n_annotated == 0:
        args.n_annotated = np.array([10, 20, 25, 30, 40, 50])

    dat, _, scaler = load_xabier_projections("rt_data", remove_non_retained=True)

    systems = ["FEM_long", "LIFE_old", "FEM_orbitrap_plasma", "RIKEN"]
    xabier_cuttoffs = [5, 1, 2, 1]

    timestamp = add_timestamp('')[1:]  # A fixed timestamp for all tests. [1:] removes the starting "-"
    for exclude_system, cuttoff in zip(systems, xabier_cuttoffs):
        print(f"********** Testing on {exclude_system} **********")
        train_data = dat[dat.system != exclude_system]
        system_data = dat[(dat.system == exclude_system) & (dat.rt_exper > cuttoff)].copy()

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
        with open(os.path.join(args.save_to, f'gp_model_excluding_{exclude_system}.pkl'), 'wb') as f:
            pickle.dump([gp, scaler], f)
        # FIXME
        continue

        for n_annotated_samples in args.n_annotated:
            results_filename = get_test_filename(args, n_annotated_samples, exclude_system, timestamp)
            support = None
            # if n_annotated_samples == 50:
            #     support = system_data[system_data.support]
            #     assert support.shape[0] == 50, 'Wait a minute, something is wrong!'
            #     support = (support.rt_pred.values.reshape(-1, 1), support.rt_exper.values)

            all_data, support_data, relative_errors, loss = meta_test(
                gp, mll, system_data, scaler, args, n_annotated_samples, support=support
            )

            all_data.to_csv(results_filename + '.csv')
            support_data.to_csv(results_filename + '_support.csv')

            # f = plt.figure(figsize=(16, 9))
            # plot_projection(all_data, support_data)
            # plt.title(exclude_system)
            # f.savefig(results_filename + '.png', bbox_inches='tight')
