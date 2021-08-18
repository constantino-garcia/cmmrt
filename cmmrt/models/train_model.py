import argparse
import pickle
from collections import namedtuple

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold

from cmmrt.models.gbm.xgboost import SelectiveXGBRegressor
from cmmrt.models.preprocessor.Preprocessors import Preprocessor
from cmmrt.models.ensemble.Blender import Blender
from cmmrt.models.gp.DKL import SkDKL
from cmmrt.models.gbm.WeightedCatBoostRegressor import WeightedCatBoostRegressor
from cmmrt.models.nn.SkDnn import SkDnn
from cmmrt.utils.data import AlvadescDataset
from cmmrt.utils.train.param_search import param_search

BlenderConfig = namedtuple('BlenderConfig', ['train_size', 'n_strats', 'random_state'])
ParamSearchConfig = namedtuple('ParamSearchConfig', ['storage', 'study_prefix', 'param_search_cv', 'n_trials'])


def create_smoke_blender(desc_cols, fgp_cols, binary_cols, blender_config):
    estimators = [
        ('desc_dkl', SkDKL(2, use_col_indices=desc_cols, binary_col_indices=binary_cols)),
        ('fgp_mlp',
         SkDnn(use_col_indices=fgp_cols, binary_col_indices=binary_cols, transform_output=True)),
    ]
    return Blender(
        estimators, RandomForestRegressor(), **blender_config._asdict()
    )


def create_blender(desc_cols, fgp_cols, binary_cols, blender_config):
    estimators = [
        # Deep Kernel Learning
        ('full_dkl', SkDKL(2, use_col_indices='all', binary_col_indices=binary_cols)),
        ('desc_dkl', SkDKL(2, use_col_indices=desc_cols, binary_col_indices=binary_cols)),
        ('fgp_dkl', SkDKL(2, use_col_indices=fgp_cols, binary_col_indices=binary_cols)),
        # Deep Neural Nets
        ('full_mlp', SkDnn(use_col_indices='all', binary_col_indices=binary_cols, transform_output=True)),
        ('desc_mlp', SkDnn(use_col_indices=desc_cols, binary_col_indices=binary_cols, transform_output=True)),
        ('fgp_mlp', SkDnn(use_col_indices=fgp_cols, binary_col_indices=binary_cols, transform_output=True)),
        # XGBoost
        ('full_xgb', SelectiveXGBRegressor(use_col_indices='all', binary_col_indices=binary_cols)),
        ('desc_xgb', SelectiveXGBRegressor(use_col_indices=desc_cols, binary_col_indices=binary_cols)),
        ('fgp_xgb', SelectiveXGBRegressor(use_col_indices=fgp_cols, binary_col_indices=binary_cols)),
        # LGBM
        ('lgb', LGBMRegressor()),
        # CatBoosters
        ('cb_0', WeightedCatBoostRegressor(1e-6, use_col_indices='all', binary_col_indices=binary_cols)),
        ('cb_05', WeightedCatBoostRegressor(0.5, use_col_indices='all', binary_col_indices=binary_cols)),
        ('cb_1', WeightedCatBoostRegressor(1., use_col_indices='all', binary_col_indices=binary_cols)),
        ('cb_20', WeightedCatBoostRegressor(20, use_col_indices='all', binary_col_indices=binary_cols)),
        ('cb_40', WeightedCatBoostRegressor(40, use_col_indices='all', binary_col_indices=binary_cols)),
        ('cb_80', WeightedCatBoostRegressor(80, use_col_indices='all', binary_col_indices=binary_cols))
    ]
    return Blender(
        estimators, RandomForestRegressor(), **blender_config._asdict()
    )


def tune_and_fit(alvadesc_data, param_search_config, blender_config, smoke_test=False):
    print("Preprocessing...")
    preprocessor = Preprocessor(
        storage=param_search_config.storage,
        study_prefix=f'preproc-{param_search_config.study_prefix}',
        desc_cols=alvadesc_data.desc_cols,
        fgp_cols=alvadesc_data.fgp_cols,
        n_trials=param_search_config.n_trials,
        search_cv=param_search_config.param_search_cv
    )
    X_train = preprocessor.fit_transform(alvadesc_data.X, alvadesc_data.y)
    features_description = preprocessor.describe_transformed_features()

    if smoke_test:
        print("Creating (smoke) blender")
        blender = create_smoke_blender(features_description['desc_cols'],
                                       features_description['fgp_cols'],
                                       features_description['binary_cols'],
                                       blender_config)
    else:
        print("Creating blender")
        blender = create_blender(features_description['desc_cols'],
                                 features_description['fgp_cols'],
                                 features_description['binary_cols'],
                                 blender_config)

    print("Param search")
    blender = param_search(
        blender,
        X_train, alvadesc_data.y,
        cv=param_search_config.param_search_cv,
        study=(param_search_config.storage, param_search_config.study_prefix),
        n_trials=param_search_config.n_trials,
        keep_going=False
    )
    print("Training")
    blender.fit(X_train, alvadesc_data.y)

    return preprocessor, blender


def create_base_parser(default_storage, default_study, description=""):
    parser = argparse.ArgumentParser(description=description)

    def restricted_float(x):
        try:
            x = float(x)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{x} not a floating-point literal")
        if x < 0.0 or x > 1.0:
            raise argparse.ArgumentTypeError(f"{x} not in range [0.0, 1.0]")
        return x

    parser.add_argument('--storage', type=str, default=default_storage,
                        help='SQLITE DB for storing the results of param search (e.g.: sqlite:///train.db')
    parser.add_argument('--study', type=str, default=default_study,
                        help='Study name to identify param search results withing the DB')
    parser.add_argument('--train_size', type=restricted_float, default=restricted_float(0.8),
                        help="Percentage of the training set to train the base classifiers. The remainder is used to "
                             "train the meta-classifier")
    parser.add_argument('--param_search_folds', type=int, default=5, help='Number of folds to be used in param search')
    parser.add_argument('--trials', type=int, default=10, help='Number of trials in param search')
    parser.add_argument('--smoke_test', action='store_true',
                        help='Use small model and subsample training data for quick testing. '
                             'param_search_folds and trials are also overriden')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for reproducibility or reusing param search results')
    return parser


def load_data_and_configs(args, download_directory):
    alvadesc_data = AlvadescDataset(download_directory)
    if args.smoke_test:
        idx = np.random.choice(np.arange(alvadesc_data.X.shape[0]), 5000, replace=False)
        alvadesc_data = alvadesc_data[idx]
        args.params_search_folds = 2
        args.trials = 2

    param_search_config = ParamSearchConfig(
        storage=args.storage,
        study_prefix=args.study,
        param_search_cv=RepeatedKFold(n_splits=args.param_search_folds, n_repeats=1, random_state=args.random_state),
        n_trials=args.trials
    )
    blender_config = BlenderConfig(
        train_size=args.train_size,
        n_strats=6,
        random_state=args.random_state + 100
    )
    return alvadesc_data, param_search_config, blender_config


if __name__ == '__main__':
    parser = create_base_parser(
        description="Train Blender", default_storage="sqlite:///../../models/optuna/train.db", default_study="train"
    )
    args = parser.parse_args()
    print(args)

    alvadesc_data, param_search_config, blender_config = load_data_and_configs(args, download_directory="../../rt_data")



    preprocessor, blender = (
        tune_and_fit(alvadesc_data, param_search_config=param_search_config, blender_config=blender_config,
                     smoke_test=args.smoke_test)
    )

    print("Saving preprocessor and blender (with base models)")
    with open("../../models/preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)
    with open("../../models/blender.pkl", "wb") as f:
        pickle.dump(blender, f)
