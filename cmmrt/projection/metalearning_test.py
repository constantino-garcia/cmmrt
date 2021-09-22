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


def meta_test(gp, mll, system_data, scaler, args, test_size, support=None):
    x = system_data.rt_pred.values.astype('float32')
    y = system_data.rt_exper.values.astype('float32')
    x = scaler.transform(x.reshape(-1, 1)).flatten()
    y = scaler.transform(y.reshape(-1, 1)).flatten()
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]
    # Note that x and y are swapped on stratified_train_test_split so that stratification is done based on x
    if support is None:
        x_support, y_support = get_representatives(x, y, test_size=test_size, n_quantiles=10)
        x_support = x_support.reshape(-1, 1)
    else:
        print('Using Xabier support')
        x_support, y_support = support
        x_support = scaler.transform(x_support)
        y_support = scaler.transform(y_support.reshape(-1, 1)).flatten()

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


def get_test_filename(args, testsize, system, timestamp):
    filename = get_basename(args)
    return filename + '-' + str(testsize) + '-' + system + '-' + timestamp


if __name__ == '__main__':
    # Create the parser
    my_parser = create_parser()
    my_parser.add_argument('-t', '--testsize', type=int, default=0)
    args = my_parser.parse_args()
    handle_saving_dir(args.save_to)
    print(args)

    if args.testsize == 0:
        args.testsize = np.array([10, 20, 25, 30, 40, 50])

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

        for testsize in args.testsize:
            results_filename = get_test_filename(args, testsize, exclude_system, timestamp)
            support = None
            # if testsize == 50:
            #     support = system_data[system_data.support]
            #     assert support.shape[0] == 50, 'Wait a minute, something is wrong!'
            #     support = (support.rt_pred.values.reshape(-1, 1), support.rt_exper.values)

            all_data, support_data, relative_errors, loss = meta_test(
                gp, mll, system_data, scaler, args, testsize, support=support
            )

            all_data.to_csv(results_filename + '.csv')
            support_data.to_csv(results_filename + '_support.csv')

            f = plt.figure(figsize=(16, 9))
            plot_projection(all_data, support_data)
            plt.title(exclude_system)
            f.savefig(results_filename + '.png', bbox_inches='tight')
