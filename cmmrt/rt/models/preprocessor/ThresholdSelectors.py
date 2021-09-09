import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn.utils.validation import check_is_fitted


class CorThreshold(SelectorMixin, BaseEstimator):
    def __init__(self, threshold=0.):
        self.threshold = threshold

    def fit(self, X, y=None):
        cor_matrix = np.abs(np.corrcoef(X, rowvar=False))
        self.upper_tri_ = np.triu(cor_matrix, k=1)
        return self

    def _get_support_mask(self):
        check_is_fitted(self)
        n = self.upper_tri_.shape[1]
        return np.array([
            all(self.upper_tri_[:column, column] < self.threshold) for column in range(n)
        ])


class MIThreshold(SelectorMixin, BaseEstimator):
    def __init__(self, threshold=0.):
        self.threshold = threshold

    def fit(self, X, y=None):
        self.mi_ = mutual_info_regression(X, y)
        self.mi_ /= np.max(self.mi_)
        return self

    def _get_support_mask(self):
        check_is_fitted(self)
        return self.mi_ > self.threshold


class FThreshold(SelectorMixin, BaseEstimator):
    def __init__(self, threshold=0.):
        self.threshold = threshold

    def fit(self, X, y=None):
        self.f_, self.pval = f_regression(X, y)
        self.nf_ = self.f_ / np.max(self.f_)
        return self

    def _get_support_mask(self):
        check_is_fitted(self)
        return self.pval > self.threshold
