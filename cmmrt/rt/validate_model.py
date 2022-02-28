"""Validate regressors on SMRT dataset

This script allows the user to train and validate a set of different regressors on the SMRT dataset
using cross-validation. Since training includes Bayesian optimization of hyperparameters,
nested cross-validation is actually used.

The models include:
* Gaussian process with deep kernels (deep kernel learning)
* Deep Neural network trained with warm restarts and Stochastic Weight averaging (SWA).
* Gradient Boosting Machines:
    * XGBoost
    * LightGBM
    * A set of CatBoost models assigning different weights to retained and non-retained molecules
* An ensemble of the previous models based on a Random Forest meta-regressors.

With the exception of the CatBoost models, the models are trained using three types of
features: 1) fingerprints, 2) descriptors and 3) fingerprints + descriptors. The
fingerprints and descriptors were obtained using the Alvadesc software.

This script permits the user to specify command line options. Use
$ python validate_model.py --help
to see the options.
"""

import os
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.model_selection import StratifiedKFold

from cmmrt.rt.data import is_non_retained
from cmmrt.rt.train_model import create_base_parser
from cmmrt.rt.train_model import load_data_and_configs
from cmmrt.rt.train_model import tune_and_fit
from cmmrt.utils.train.model_selection import stratify_y


def create_cv_parser(default_storage, default_study, description):
    """Command line parser for validating all models"""
    parser = create_base_parser(default_storage=default_storage, default_study=default_study, description=description)
    parser.add_argument("--cv_folds", type=int, default=5, help="Number of folds to be used for CV")
    parser.add_argument("--csv_output", type=str, default=os.path.join(tempfile.gettempdir(), "cv_results.csv"),
                        help="CSV file to store the CV results")
    return parser


def evaluate_all_estimators(blender, X_test, y_test, metrics, fold_number):
    """Evaluate all estimators in blender on the test set"""
    is_non_retained_indicator = is_non_retained(y_test)
    molecules_indices = {
        'all': (np.ones_like(is_non_retained_indicator) == 1),
        'retained': (is_non_retained_indicator == 0),
        'non-retained': (is_non_retained_indicator == 1)
    }
    iteration_results = []
    for estimator_name, estimator in blender._fitted_estimators + [('Blender', blender)]:
        for molecules, indices in molecules_indices.items():
            estimator_results = {k: metric(y_test[indices], estimator.predict(X_test[indices, ...])) for k, metric in metrics.items()}
            estimator_results['estimator'] = estimator_name
            estimator_results['fold'] = fold_number
            estimator_results['molecules'] = molecules
            iteration_results.append(estimator_results)
    return pd.DataFrame(iteration_results)


if __name__ == '__main__':
    parser = create_cv_parser(
        description="Cross-validate Blender", default_storage="sqlite:///cv.db", default_study="cv"
    )
    args = parser.parse_args()

    alvadesc_data, param_search_config, blender_config = load_data_and_configs(args, download_directory="rt_data")

    cv_folds = 2 if args.smoke_test else args.cv_folds
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=args.random_state + 500)
    strats = stratify_y(alvadesc_data.y, blender_config.n_strats)

    metrics = {'mae': mean_absolute_error, 'medae': median_absolute_error}

    base_study_name = param_search_config.study_prefix
    results = []
    for fold, (train_index, test_index) in enumerate(cv.split(alvadesc_data.X, strats)):
        # FIXME: should replace namedTuples with recordtype to avoid next line
        param_search_config = param_search_config._replace(study_prefix=base_study_name + f"-fold-{fold}")
        alvadesc_train = alvadesc_data[train_index]
        alvadesc_test = alvadesc_data[test_index]

        preprocessor, blender = (
            tune_and_fit(alvadesc_train, param_search_config=param_search_config, blender_config=blender_config,
                         smoke_test=args.smoke_test)
        )

        X_test = preprocessor.transform(alvadesc_test.X)
        results.append(
            evaluate_all_estimators(blender, X_test, alvadesc_test.y, metrics, fold)
        )
    results = pd.concat(results, axis=0)

    print(f"Saving results to {args.csv_output}")
    print(results)
    results.to_csv(args.csv_output, index=False)