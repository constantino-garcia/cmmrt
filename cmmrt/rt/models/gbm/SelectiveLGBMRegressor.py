import inspect

import optuna
from optuna.trial import TrialState
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMRegressor

from cmmrt.rt.models.base.PipelineWrapper import RTRegressor


class SelectiveLGBMRegressor(RTRegressor):
    def __init__(self, boosting_type='gbdt', num_leaves=31, max_depth=- 1, learning_rate=0.1, n_estimators=100,
                 subsample_for_bin=200000, objective=None, class_weight=None, min_split_gain=0.0, min_child_weight=0.001,
                 min_child_samples=20, subsample=1.0, subsample_freq=0, colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0,
                 random_state=None, n_jobs=- 1, importance_type='split',
                 use_col_indices='all', binary_col_indices=None, var_p=0, transform_output=False):
        super().__init__(use_col_indices, binary_col_indices, var_p, transform_output)
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)

    def _init_regressor(self):
        return LGBMRegressor(**self._rt_regressor_params())
