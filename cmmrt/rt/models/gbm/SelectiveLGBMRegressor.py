import inspect

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, BaseEstimator


class SelectiveLGBMRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, boosting_type='gbdt', num_leaves=31, max_depth=- 1, learning_rate=0.1, n_estimators=100,
                 subsample_for_bin=200000, objective=None, class_weight=None, min_split_gain=0.0, min_child_weight=0.001,
                 min_child_samples=20, bagging_fraction=1.0, bagging_freq=0, feature_fraction=1.0, lambda_l1=0.0, lambda_l2=0.0,
                 random_state=None, n_jobs=- 1, importance_type='split',
                 metric="l1", verbosity=-1, feature_pre_filter=True,
                 use_col_indices='all', binary_col_indices=None):
        super().__init__()
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)

    def _select_columns(self, X):
        X_df = pd.DataFrame(X)
        if isinstance(self.use_col_indices, str):
            assert self.use_col_indices == 'all', "Invalid use_col_indices"
            X_df[self.binary_col_indices] = X_df[self.binary_col_indices].astype('category')
            return X_df, self.binary_col_indices
        else:
            binary_indices = np.array([idx in self.binary_col_indices for idx in np.arange(X.shape[1])])
            binary_indices = np.where(binary_indices[self.use_col_indices])[0]
            X_df[binary_indices] = X_df[binary_indices].astype('category')
            return X_df[self.use_col_indices], list(binary_indices.astype('int'))

    def to_lgb_dataset(self, X_, y):
        X, binary_indices = self._select_columns(X_)
        return lgb.Dataset(X, label=y)

    def _init_regressor(self):
        params = self.get_params().copy()
        params.pop("use_col_indices")
        params.pop("binary_col_indices")
        return lgb.LGBMRegressor(**params)

    def fit(self, X_, y):
        self._model = self._init_regressor()
        X, binary_indices = self._select_columns(X_)
        self._model.fit(X, y)
        return self

    def predict(self, X_):
        X, _ = self._select_columns(X_)
        return self._model.predict(X)