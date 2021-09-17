import inspect

from catboost import CatBoostRegressor
from catboost import Pool
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin

from cmmrt.rt.models.base.PipelineWrapper import RTRegressor
from cmmrt.rt.data import is_non_retained


class _CatBoostRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, nr_weight):
        super().__init__()
        self.nr_weight = nr_weight

    def fit(self, X, y):
        def assign_weights(y, non_retained_weight, retained_weight):
            non_retained = is_non_retained(y)
            return non_retained_weight * non_retained + retained_weight * (1 - non_retained)

        def weight_function(y):
            return assign_weights(y, self.nr_weight, 1)

        self._model = CatBoostRegressor().fit(
            Pool(
                data=X,
                label=y,
                weight=weight_function(y)
            )
        )
        return self

    def predict(self, X):
        return self._model.predict(X)


class WeightedCatBoostRegressor(RTRegressor):
    def __init__(self, nr_weight, use_col_indices='all', binary_col_indices=None, var_p=0, transform_output=False):
        super().__init__(use_col_indices, binary_col_indices, var_p, transform_output)
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)

    def _init_regressor(self):
        return _CatBoostRegressor(nr_weight=self.nr_weight)
