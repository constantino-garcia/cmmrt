"""Meta-train GP on the PredRet dataset

This script allows the user to meta-train a GP model on the PredRet dataset as done in the paper
<<Domingo-Almenara, Xavier, et al. "The METLIN small molecule dataset for machine learning-based retention
time prediction." Nature communications 10.1 (2019): 1-9>>
(that is, predicted RTs are taken from this paper).

Meta-training is done by using the GP model to create projections between the predicted RTs and the experimental
RTs measured using different chromatographic methods. The result of meta-training is a GP with tuned hyperparameters
that can be interpreted as a sensible prior to be used when creating a projection model from a small amount of data.

This script permits the user to specify command line options. Use
$ python metalearning_train.py --help
to see the options.
"""

import os
import warnings

import torch

from cmmrt.projection.data import _load_predret_with_xabier_predictions, load_predret_with_predictions
from cmmrt.projection.projection_tasks import ProjectionsTasks
from cmmrt.projection.metatrainers.utils import create_metatrainer
from cmmrt.projection.models.utils import create_projector_and_optimizer
from cmmrt.utils.generic_utils import handle_saving_dir

warnings.simplefilter("ignore")


def create_parser():
    """Command line parser for the meta-training and meta-testing scripts."""
    import argparse
    my_parser = argparse.ArgumentParser(description='meta-train gp on projection tasks')
    my_parser.add_argument('--dataset', type=str, default='cmm')
    my_parser.add_argument('--direction', type=str, default='p2e',
                           help='p2e (predicted to experimental) or e2p (experimental to predicted)')
    my_parser.add_argument('--epochs', type=int, default=-1,
                           help='Number of epochs for meta-train the gp. Use -1 '
                                'to denote "until convergence"')
    my_parser.add_argument('--inner_epochs', type=int, default=10)
    my_parser.add_argument('--device', type=str, default='cpu')
    my_parser.add_argument('--mean', type=str, default='zero')  # mlp_mean or not
    my_parser.add_argument('--kernel', type=str, default='rbf+linear')
    my_parser.add_argument('--save_to', type=str, default='.', help='folder where to save the figures/results')
    my_parser.add_argument('--weight_decay', type=float, default=4e-3)
    my_parser.add_argument('--seed', type=int, default=0)

    return my_parser


def create_meta_models(args):
    projector, inner_optimizer = create_projector_and_optimizer(
        model_type='exact',
        mean=args.mean,
        kernel=args.kernel,
        lr=1e-3,
        weight_decay=args.weight_decay,
        device=args.device
    )
    metatrainer = create_metatrainer(
        metatrainer_name='naive',
        projector=projector,
        inner_optimizer=inner_optimizer,
        inner_epochs=args.inner_epochs,
        outer_epochs=args.epochs,
        device=args.device
    )
    return projector, inner_optimizer, metatrainer


def get_basename(args):
    """Use command line arguments to create a basename for the results"""
    filename = args.direction + "_"
    filename += args.mean
    filename += ('_' + args.kernel)
    filename += "_{:.0e}".format(args.weight_decay).replace('-', '')
    filename += f"_{args.epochs}_{args.inner_epochs}"
    filename = os.path.join(args.save_to, filename)
    return filename


def add_timestamp(filename):
    """Add timestamp to filename"""
    import time
    timestamp = str(int(round(time.time() * 1000)))
    return filename + f"-{timestamp}"


def split_systems_on_train_test(data, direction, x_scaler, y_scaler,
                                systems=["FEM_long", "LIFE_old", "FEM_orbitrap_plasma", "RIKEN"]):
    train_data = data[~data.system.isin(systems)]
    test_data = data[data.system.isin(systems)]
    train_tasks = ProjectionsTasks(train_data, direction, (1.0, 1.0), min_n=20, x_scaler=x_scaler, y_scaler=y_scaler)
    if test_data.empty:
        test_tasks = None
    else:
        test_tasks = ProjectionsTasks(test_data, direction, (1.0, 1.0), min_n=20, x_scaler=x_scaler, y_scaler=y_scaler)
    return train_tasks, test_tasks


def load_train_test_tasks(dataset, direction, download_directory, remove_non_retained, test_systems):
    data, systems, x_scaler, y_scaler = load_predret_with_predictions(dataset, download_directory, remove_non_retained)
    train_tasks, test_tasks = split_systems_on_train_test(data, direction, x_scaler, y_scaler, test_systems)
    return train_tasks, test_tasks


if __name__ == '__main__':
    # Create the parser
    my_parser = create_parser()
    args = my_parser.parse_args()
    handle_saving_dir(args.save_to)

    train_tasks, test_tasks = load_train_test_tasks(
        args.dataset,
        args.direction,
        download_directory="rt_data",
        remove_non_retained=True,
        test_systems=['']
    )

    projector, inner_optimizer, metatrainer = create_meta_models(args)
    projector.prepare_metatraining()
    metatrainer.metatrain(train_tasks)
    torch.save(projector.state_dict(), get_basename(args))
    print('Done meta-training')
