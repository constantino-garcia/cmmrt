import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler, FunctionTransformer


class RTTransformer(BaseEstimator, TransformerMixin):
    """Transformer that combines a logarithmic transformation log(1 + x) with a RobustScaler."""

    def __init__(self, transformation=""):
        assert transformation in ['normalization', 'standardization']
        self.transformation = transformation

    def fit(self, X, y=None):
        self._scaler = RobustScaler(with_centering=True, unit_variance=True)
        self._scaler.fit(np.log(1 + X))
        if self.transformation == 'normalization':
            self._postproc = FunctionTransformer(lambda x: (x + 3) / 6, inverse_func=lambda x: x * 6 - 3)
        else:
            self._postproc = FunctionTransformer(lambda x: x / 3, inverse_func=lambda x: x * 3)
        return self

    def transform(self, X):
        try:
            return self._postproc.transform(self._scaler.transform(np.log(1 + X)))
        except Exception as e:
            print(e)

    def inverse_transform(self, X):
        x_ = self._scaler.inverse_transform(self._postproc.inverse_transform(X))
        return np.exp(x_) - 1

    def inverse_ci(self, mean, var, z=1.96):
        new_mean = self._scaler.inverse_transform(
            self._postproc.inverse_transform(mean.reshape(-1, 1))
        ).flatten()
        if self.transformation == 'normalization':
            postproc_scale = 6
        else:
            postproc_scale = 3
        new_var = var * (postproc_scale ** 2) * (self._scaler.scale_ ** 2)
        geometric_mean = np.exp(new_mean)
        geometric_interval = np.exp(z * np.sqrt(new_var))
        lb_tmp = geometric_mean / geometric_interval
        ub_tmp = geometric_mean * geometric_interval
        lb = np.minimum(lb_tmp, ub_tmp)
        ub = np.maximum(lb_tmp, ub_tmp)
        median = geometric_mean
        new_mean = geometric_mean * np.exp(0.5 * new_var)
        new_mean -= 1
        median -= 1
        lb -= 1
        ub -= 1
        return new_mean, median, lb, ub
