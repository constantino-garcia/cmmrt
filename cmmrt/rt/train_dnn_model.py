import argparse
import pickle
from collections import namedtuple

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_absolute_error, median_absolute_error

from cmmrt.rt.data import AlvadescDataset, RDkitDataset
from cmmrt.rt.models.nn.SkDnn import SkDnn
from cmmrt.rt.models.preprocessor.Preprocessors import FgpPreprocessor
from cmmrt.utils.generic_utils import handle_saving_dir
from cmmrt.utils.train.model_selection import stratified_train_test_split
from cmmrt.utils.train.param_search import param_search
from cmmrt.utils.train.param_search import create_study

ParamSearchConfig = namedtuple('ParamSearchConfig', ['storage', 'study_prefix', 'param_search_cv', 'n_trials'])


def create_dnn(fgp_cols, binary_cols):
    return SkDnn(use_col_indices=fgp_cols, binary_col_indices=binary_cols, transform_output=True)


def tune_and_fit(data, param_search_config):
    print(f"Starting tune_and_fit with data with dim ({data.X.shape[0]},{data.X.shape[1]})")
    print("Preprocessing...")
    preprocessor = FgpPreprocessor(
        storage=param_search_config.storage,
        study_prefix=f'preproc-{param_search_config.study_prefix}',
        fgp_cols=data.fgp_cols,
        n_trials=param_search_config.n_trials,
        search_cv=param_search_config.param_search_cv
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
    data = RDkitDataset(download_directory)
    print(args.smoke_test)
    if args.smoke_test:
        idx = np.random.choice(np.arange(data.X.shape[0]), 5000, replace=False)
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
    parser = create_train_parser(default_storage="sqlite:///rdkit_dnn.db", default_study="dnn")
    args = parser.parse_args()
    handle_saving_dir(args.save_to)

    data, param_search_config = load_data_and_configs(args, download_directory="rt_data")
    print(args)

    # TODO: This is a quick test due to deadline restrictions. Hence, we test the performance on a hold-out set. In
    # the future, should be done with validate_dnn_model
    train_indices, test_indices, _, _ = stratified_train_test_split(np.arange(data.X.shape[0]), data.y,
                                                                    test_size=0.2, n_strats=6, random_state=args.random_state)
    train_data = data[train_indices]
    test_data = data[test_indices]

    preprocessor, dnn = (
        tune_and_fit(train_data, param_search_config=param_search_config)
    )

    metrics = {'mae': mean_absolute_error, 'medae': median_absolute_error}
    predictions = dnn.predict(preprocessor.transform(test_data.X))
    performance = {k: metric(test_data.y, predictions) for k, metric in metrics.items()}

    print("Saving preprocessor, DNN, and metrics")
    pd.DataFrame(performance, index=[0]).to_csv("rdkit_dnn_performance.csv")
    with open(os.path.join(args.save_to, "preprocessor.pkl"), "wb") as f:
        pickle.dump(preprocessor, f)
    with open(os.path.join(args.save_to, "dnn.pkl"), "wb") as f:
        pickle.dump(dnn, f)
