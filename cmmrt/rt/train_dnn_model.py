import argparse
import pickle
from collections import namedtuple

import numpy as np
from sklearn.model_selection import RepeatedKFold

from cmmrt.rt.data import AlvadescDataset, PredRetFeaturizedSystem
from cmmrt.rt.models.nn.SkDnn import SkDnn
from cmmrt.rt.models.preprocessor.Preprocessors import FgpPreprocessor
from cmmrt.utils.generic_utils import handle_saving_dir
from cmmrt.utils.train.param_search import param_search
from cmmrt.utils.train.param_search import create_study

ParamSearchConfig = namedtuple('ParamSearchConfig', ['storage', 'study_prefix', 'param_search_cv', 'n_trials'])


def create_dnn(fgp_cols, binary_cols):
    return SkDnn(use_col_indices=fgp_cols, binary_col_indices=binary_cols, transform_output=True)


def tune_and_fit(data, param_search_config):
    print(f"Starting tune_and_fit with data with dim ({data.X.shape[0]},{data.X.shape[1]})")
    print(f"Preprocessing {data.name}...")
    n_trials = param_search_config.n_trials
    if data.name == "alvadesc":
        non_retained_cutoff = 300
    elif "FEM_long" in data.name:
        non_retained_cutoff = 5
    elif "LIFE_old" in data.name:
        non_retained_cutoff = 1
    elif "FEM_orbitrap" in data.name:
        non_retained_cutoff = 2
    elif "RIKEN" in data.name:
        non_retained_cutoff = 1
    else:
        # No trials since there is no definition for non-retained molecules in this system
        n_trials = 0
        non_retained_cutoff = 0
    print(f"Non-retained cutoff: {non_retained_cutoff}")

    preprocessor = FgpPreprocessor(
        storage=param_search_config.storage,
        study_prefix=f'preproc-{param_search_config.study_prefix}',
        fgp_cols=data.fgp_cols,
        n_trials=n_trials,
        search_cv=param_search_config.param_search_cv,
        non_retained_cutoff=non_retained_cutoff
    )
    X_train = preprocessor.fit_transform(data.X, data.y)

    print("Creating DNN")
    all_cols = np.arange(X_train.shape[1])
    dnn = create_dnn(fgp_cols=all_cols, binary_cols=all_cols[:-1])

    print("Param search")
    study = create_study("dnn", param_search_config.study_prefix, param_search_config.storage)
    best_params = param_search(
        dnn,
        X_train, data.y,
        cv=param_search_config.param_search_cv,
        study=study,
        n_trials=param_search_config.n_trials,
        keep_going=False
    )
    print("Training")
    dnn = create_dnn(fgp_cols=all_cols, binary_cols=all_cols[:-1])
    dnn.set_params(**best_params)
    dnn.fit(X_train, data.y)

    return preprocessor, dnn


def create_train_parser(default_storage, default_study):
    parser = argparse.ArgumentParser(description="Train DNN")
    parser.add_argument("--dataset", type=str, default="alvadesc", help="Dataset to use")
    parser.add_argument("--cutoff", type=float, default=0.0, help="Cut-off value to consider a molecule as retained."
                                                                  " Non-retained molecules will be eliminated.")
    parser.add_argument('--storage', type=str, default=default_storage,
                        help='SQLITE DB for storing the results of param search (e.g.: sqlite:///train.db')
    parser.add_argument('--study', type=str, default=default_study,
                        help='Study name to identify param search results withing the DB')
    parser.add_argument('--param_search_folds', type=int, default=5, help='Number of folds to be used in param search')
    parser.add_argument('--trials', type=int, default=10, help='Number of trials in param search')
    parser.add_argument('--smoke_test', action='store_true',
                        help='Use small model and subsample training data for quick testing. '
                             'param_search_folds and trials are also overridden')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for reproducibility or reusing param search results')
    parser.add_argument('--save_to', type=str, default='.', help='folder where to save the preprocessor and regressor models')
    return parser


def load_data_and_configs(args, download_directory):
    if args.dataset == "alvadesc":
        print("Loading Alvadesc dataset")
        data = AlvadescDataset(download_directory)
    else:
        print(f"Loading {args.dataset} dataset")
        print(f"-----------------> {args.cutoff}")
        data = PredRetFeaturizedSystem(args.dataset, args.cutoff, download_directory)
        if data.X.shape[0] == 0:
            raise ValueError(f"Dataset {args.dataset} is empty")
    print(data.X.shape[0])
    print(args.smoke_test)
    if args.smoke_test:
        idx = np.random.choice(np.arange(data.X.shape[0]), min(5000, data.X.shape[0] // 3), replace=False)
        data = data[idx]
        args.param_search_folds = 2
        args.trials = 2

    param_search_config = ParamSearchConfig(
        storage=args.storage,
        study_prefix=args.study,
        param_search_cv=RepeatedKFold(n_splits=args.param_search_folds, n_repeats=1, random_state=args.random_state),
        n_trials=args.trials
    )
    return data, param_search_config


if __name__ == '__main__':
    import os
    parser = create_train_parser(default_storage="sqlite:///dnn.db", default_study="dnn")
    args = parser.parse_args()
    handle_saving_dir(args.save_to)

    data, param_search_config = load_data_and_configs(args, download_directory="rt_data")
    print(args)

    preprocessor, dnn = (
        tune_and_fit(data, param_search_config=param_search_config)
    )

    print("Saving preprocessor and DNN")
    with open(os.path.join(args.save_to, "preprocessor.pkl"), "wb") as f:
        pickle.dump(preprocessor, f)
    with open(os.path.join(args.save_to, "dnn.pkl"), "wb") as f:
        pickle.dump(dnn, f)
