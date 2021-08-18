from abc import ABCMeta
from abc import abstractmethod

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer


class PipelineWrapper(BaseEstimator, metaclass=ABCMeta):
    @abstractmethod
    def _init_model(self) -> Pipeline:
        pass

    @abstractmethod
    def _translate_params(self) -> dict:
        pass

    def fit(self, X, y):
        self._model = self._init_model().set_params(**self._translate_params())
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)


class RTRegressor(PipelineWrapper, RegressorMixin, metaclass=ABCMeta):
    def __init__(self, use_col_indices, binary_col_indices, var_p, transform_output):
        self.use_col_indices = use_col_indices
        self.binary_col_indices = binary_col_indices
        self.var_p = var_p
        self.transform_output = transform_output

    @abstractmethod
    def _init_regressor(self):
        pass

    def _init_model(self):
        if isinstance(self.use_col_indices, str) and self.use_col_indices == 'all':
            relevant_binary_indices = self.binary_col_indices
            preproc = ColumnTransformer(
                [("var_th", VarianceThreshold(self.var_p * (1 - self.var_p)), relevant_binary_indices)],
                remainder='passthrough'
            )
        else:
            relevant_binary_indices = np.intersect1d(self.use_col_indices, self.binary_col_indices)
            relevant_non_binary_indices = np.setdiff1d(self.use_col_indices, self.binary_col_indices)
            preproc = ColumnTransformer(
                [('keep', 'passthrough', relevant_non_binary_indices),
                 ('var_th', VarianceThreshold(self.var_p * (1 - self.var_p)), relevant_binary_indices)]
            )
        base_pipeline = Pipeline([
            ('preproc', preproc),
            ('regressor', self._init_regressor())
        ])
        if self.transform_output:
            transformer = QuantileTransformer(n_quantiles=1000, output_distribution='normal')
        else:
            transformer = None
        return TransformedTargetRegressor(regressor=base_pipeline, transformer=transformer)

    def _rt_regressor_params(self):
        params = self.get_params().copy()
        params.pop('use_col_indices')
        params.pop('var_p')
        params.pop('transform_output')
        params.pop('binary_col_indices')
        return params

    def _translate_params(self) -> dict:
        regressor_pars = self._rt_regressor_params().copy()
        prefix = 'regressor__regressor__'
        regressor_pars = {prefix + k: v for k, v in regressor_pars.items()}
        regressor_pars.update({
            'regressor__preproc__var_th__threshold': self.var_p * (1 - self.var_p),
        })
        return regressor_pars
