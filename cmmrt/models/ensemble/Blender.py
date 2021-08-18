import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from cmmrt.utils.train.model_selection import stratify_y


class Blender(BaseEstimator, RegressorMixin):
    def __init__(self, estimators, final_estimator, train_size, n_strats, random_state=42):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.train_size = train_size
        self.n_strats = n_strats
        self.random_state = random_state

    def _blending_split(self, X, y):
        codes = self.blending_strats(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=self.train_size, random_state=self.random_state,
            stratify=codes
        )
        return X_train, X_test, y_train, y_test

    def blending_strats(self, y):
        return stratify_y(y, self.n_strats)

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = self._blending_split(X, y)
        blended_X = np.concatenate(
            [clone(model).fit(X_train, y_train).predict(X_test).reshape(-1, 1) for _, model in self.estimators],
            axis=1
        )
        self._fitted_estimators = [
            (model_name, clone(model).fit(X, y)) for model_name, model in self.estimators
        ]
        self.final_estimator = clone(self.final_estimator).fit(blended_X, y_test)
        return self

    def predict(self, X):
        blended_X = np.concatenate(
            [model.predict(X).reshape(-1, 1) for _, model in self._fitted_estimators],
            axis=1
        )
        return self.final_estimator.predict(blended_X)
