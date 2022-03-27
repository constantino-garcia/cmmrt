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
import pickle
import warnings
from copy import deepcopy

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyro.optim
import torch
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import median_absolute_error, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from torch.utils.data import DataLoader
from tqdm import tqdm

from cmmrt.projection.data import get_representatives
from cmmrt.projection.metalearning_train import (
    create_parser, get_basename, add_timestamp, create_meta_models
)
from cmmrt.projection.metalearning_train import load_train_test_tasks
from cmmrt.projection.models.projector.gp_projector import GPProjector
from cmmrt.projection.models.utils import create_projector_and_optimizer
from cmmrt.utils.generic_utils import handle_saving_dir

warnings.simplefilter("ignore")


def plot_projection(proj_data, support_data):
    """Plot the projections learnt using the support data."""
    proj_data.sort_values(by=['x'], inplace=True)
    support_data.sort_values(by=['x'], inplace=True)
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


def get_test_basename(args):
    filename = get_basename(args)
    filename = filename + "_" + str(args.finetuning_epochs)
    return filename


def get_test_filename(args, n_annotated_samples, system, timestamp):
    """Get a filename to be used to store testing results based on the command line arguments and the system name."""
    filename = get_test_basename(args)
    return filename + '-' + str(n_annotated_samples) + '-' + system + '-' + timestamp


def create_test_parser():
    my_parser = create_parser()
    my_parser.add_argument('-t', '--n_annotated', type=int, default=0,
                           help='Number of samples to use for creating the projection function.')
    my_parser.add_argument('-n', '--n_repeats', type=int, default=10,
                           help='Number of repetions to use for evaluation.')
    my_parser.add_argument('--finetuning_epochs', type=int, default=500)
    my_parser.add_argument('--checkpoint', type=str, default=None)
    my_parser.add_argument('--test_systems', nargs="+", type=str,
                           default=["FEM_long", "LIFE_old", "FEM_orbitrap_plasma", "RIKEN"])
    args = my_parser.parse_args()
    handle_saving_dir(args.save_to)
    if args.n_annotated == 0:
        args.n_annotated = np.array([10, 20, 25, 30, 40, 50])
    return args


def metatest(test_tasks, weights_before, model_optimizer_factory, args, finetuning_epochs=10,
             save_partial_results=True, do_plots=True):
    """" we use a factory since pyro has trouble reseting the param store"""
    tasks_dl = DataLoader(test_tasks, batch_size=1, shuffle=False)
    if isinstance(args.n_annotated, int):
        args.n_annotated = [args.n_annotated]
    all_results = []

    def create_seed_generator():
        # print("------------------ New generator ----------------------------")
        np.random.seed(args.seed)
        M = np.iinfo(np.int32).max
        while True:
            seed = np.random.randint(M, size=1)[0]
            # print(seed, end=', ')
            yield seed

    for x, y, system in tasks_dl:
        system = system[0]
        # get rid of the batch dimension
        x = x.squeeze(0)
        y = y.squeeze(0)
        for n_annotated_samples in args.n_annotated:
            # Create a random seed generator starting from the same seed, so that different runs always use
            # the same training points for comparison.
            seed_generator = create_seed_generator()
            print('==============================================')
            print('====== system: {}, n_annotated: {} ======'.format(system, n_annotated_samples))
            print('==============================================')
            for repeat_number in range(args.n_repeats):
                timestamp = add_timestamp('')[1:]
                x_support, y_support, x_test, y_test = get_representatives(
                    x, y, test_size=n_annotated_samples, n_quantiles=10,
                    random_state_generator=seed_generator,
                    return_complement=True
                )

                x_support, y_support = torch.from_numpy(x_support).float(), torch.from_numpy(y_support).float()
                x_test, y_test = torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float()
                if args.device == 'cuda':
                    x = x.cuda()
                    y = y.cuda()
                    x_support = x_support.cuda()
                    y_support = y_support.cuda()
                    x_test = x_test.cuda()
                    y_test = y_test.cuda()

                projector, _ = model_optimizer_factory()
                assert isinstance(projector, GPProjector), "Currently only GPProjector is supported"
                projector.load_state_dict(weights_before)
                projector.set_train_data(x_support, y_support, strict=False)
                projector.prepare_metatesting()

                inner_optimizer = torch.optim.Adam(projector.parameters(), lr=0.01)
                iteration_bar = tqdm(range(finetuning_epochs))
                projector.train()
                for _ in iteration_bar:
                    loss = projector.train_step(x_support, y_support, inner_optimizer)
                    if isinstance(loss, torch.Tensor):
                        loss = loss.item()
                    iteration_bar.set_postfix(finetuning_loss=loss)

                projector.eval()
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(projector.likelihood, projector.gp)
                projector.eval()
                with torch.no_grad():
                    mll_value = mll(projector(x_test), y_test).item()

                with torch.no_grad():
                    int_dist = projector(x)
                    pred_dist = projector.likelihood(int_dist)
                mean, var = int_dist.mean.cpu().numpy(), int_dist.variance.cpu().numpy()
                mean_pred, var_pred = pred_dist.mean.cpu().numpy(), pred_dist.variance.cpu().numpy()

                with torch.no_grad():
                    int_dist_test = projector(x_test)
                    pred_dist_test = projector.likelihood(int_dist_test)
                mean_test, var_test = int_dist_test.mean.cpu().numpy(), int_dist_test.variance.cpu().numpy()
                mean_pred_test, var_pred_test = pred_dist_test.mean.cpu().numpy(), pred_dist_test.variance.cpu().numpy()

                x_, y_ = x.cpu().numpy(), y.cpu().numpy()
                x_support_, y_support_ = x_support.cpu().numpy(), y_support.cpu().numpy()
                x_test_, y_test_ = x_test.cpu().numpy(), y_test.cpu().numpy()

                x_, y_ = test_tasks.inverse_transform(x_, y_)
                x_support_, y_support_ = test_tasks.inverse_transform(x_support_, y_support_)
                x_test_, y_test_ = test_tasks.inverse_transform(x_test_, y_test_)
                new_mean, median, lb, ub = test_tasks.inverse_ci(mean, var)
                pred_mean, pred_median, pred_lb, pred_ub = test_tasks.inverse_ci(mean_pred, var_pred)
                new_mean_test, median_test, lb_test, ub_test = test_tasks.inverse_ci(mean_test, var_test)
                pred_mean_test, pred_median_test, pred_lb_test, pred_ub_test = test_tasks.inverse_ci(mean_pred_test,
                                                                                                     var_pred_test)

                all_data, support_data, non_support_data = pack_data(
                    x_support_, y_support_, x_test_, y_test_, x_, y_,
                    new_mean, median, lb, ub,
                    pred_mean, pred_median, pred_lb, pred_ub,
                    new_mean_test, median_test, lb_test, ub_test,
                    pred_mean_test, pred_median_test, pred_lb_test, pred_ub_test
                )

                metrics = evaluate_performance(all_data, support_data, non_support_data, mll_value)
                metrics = add_metadata(metrics, system, n_annotated_samples, repeat_number, timestamp, args)
                all_results.append(metrics)

                log_results(support_data, non_support_data, all_data, system, n_annotated_samples,
                            timestamp, args, save_partial_results, do_plots)
    print('done!')
    return pd.DataFrame(all_results)


def log_results(support_data, non_support_data, all_data, system, n_annotated_samples, timestamp, args,
                save_partial_results, do_plots):
    results_filename = get_test_filename(args, n_annotated_samples, system, timestamp)
    if save_partial_results:
        all_data.to_csv(results_filename + '.csv')
        support_data.to_csv(results_filename + '_support.csv')
        non_support_data.to_csv(results_filename + '_nonsupport.csv')
    if do_plots:
        f = plt.figure(figsize=(16, 9))
        plot_projection(all_data, support_data)
        plt.title(system)
        f.savefig(results_filename + '.png', bbox_inches='tight')


def add_metadata(metrics, system, n_annotated_samples, repeat_number, timestamp, args):
    metrics['system'] = system
    metrics['mean'] = args.mean
    metrics['kernel'] = args.kernel
    metrics['n_annotated'] = n_annotated_samples
    metrics['system'] = system
    metrics['repetition'] = repeat_number
    metrics['timestamp'] = timestamp
    return metrics


def evaluate_performance(all_data, support_data, non_support_data, mll_value):
    # errors without support
    # non_support_data = all_data.merge(support_data, how='outer', indicator=True, on=['x', 'y'])
    # non_support_data = non_support_data[non_support_data._merge != "both"]

    def robust_pol_medae(x_support, y_support, x, y):
        pm = make_pipeline(RobustScaler(), PolynomialFeatures(4), HuberRegressor())
        pm.fit(x_support, y_support)
        predictions = pm.predict(x)
        return median_absolute_error(y, predictions)

    def relative_error(y, y_pred):
        return np.median(np.abs(y - y_pred) / y)

    y = non_support_data[['y']].values.flatten()
    mean = non_support_data[['mean']].values.flatten()
    median = non_support_data[['med']].values.flatten()

    medae_value = median_absolute_error(y, mean)
    metrics = {
        'mae': mean_absolute_error(y, mean),
        'medae': medae_value,
        'mae_med': mean_absolute_error(y, median),
        'medae_med': median_absolute_error(y, median),
        'relative_error': relative_error(y, mean),
        'comp_medae': medae_value / robust_pol_medae(support_data[['x']].values,
                                                     support_data[['y']].values.flatten(),
                                                     non_support_data[['x']].values,
                                                     non_support_data[['y']].values.flatten()),
        'mll': mll_value
    }
    return metrics


def pack_data(x_support, y_support, x_non_support, y_non_support, x, y, new_mean, median, lb, ub, pred_mean,
              pred_median, pred_lb, pred_ub,
              new_mean_test, median_test, lb_test, ub_test,
              pred_mean_test, pred_median_test, pred_lb_test, pred_ub_test,
              ):
    all_data = pd.DataFrame({
        'x': x.flatten(),
        'y': y,
        'mean': new_mean,
        'med': median,
        'lb': lb,
        'ub': ub,
        'pred_mean': pred_mean,
        'pred_med': pred_median,
        'pred_lb': pred_lb,
        'pred_ub': pred_ub
    })
    support_data = pd.DataFrame({
        'x': x_support.flatten(),
        'y': y_support,
    })
    non_support_data = pd.DataFrame({
        'x': x_non_support.flatten(),
        'y': y_non_support,
        'mean': new_mean_test,
        'med': median_test,
        'lb': lb_test,
        'ub': ub_test,
        'pred_mean': pred_mean_test,
        'pred_med': pred_median_test,
        'pred_lb': pred_lb_test,
        'pred_ub': pred_ub_test
    })

    return all_data, support_data, non_support_data


if __name__ == '__main__':
    # Create the parser
    args = create_test_parser()

    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)

    train_tasks, test_tasks = load_train_test_tasks(
        args.dataset,
        args.direction,
        download_directory="rt_data",
        remove_non_retained=True,
        test_systems=args.test_systems
    )

    projector, inner_optimizer, metatrainer = create_meta_models(args)

    if args.checkpoint is not None:
        print('Loading meta-training session from {}'.format(get_test_basename(args)))
        projector.load_state_dict(torch.load(args.checkpoint))
    else:
        projector.prepare_metatraining()
        metatrainer.metatrain(train_tasks)
        print('Done metatraining. Saving to {}'.format(get_test_basename(args)))
        torch.save(projector.state_dict(), get_test_basename(args))


    def model_optimizer_factory():
        return create_projector_and_optimizer(
            model_type='exact',
            mean=args.mean,
            kernel=args.kernel,
            lr=1e-3,
            weight_decay=args.weight_decay,
            device=args.device
        )


    weights_before = deepcopy(projector.state_dict())  # save snapshot before evaluation
    results = metatest(test_tasks, weights_before, model_optimizer_factory, deepcopy(args),
                       finetuning_epochs=args.finetuning_epochs)
    print(results)
    results.to_csv(get_test_basename(args) + '_metaeval_summary.csv')
