import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import f_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cmmrt.rt.models.gbm.xgboost import create_clf
from cmmrt.rt.models.gbm.xgboost import train_clf
from cmmrt.rt.models.preprocessor.ThresholdSelectors import CorThreshold
from cmmrt.rt.data import binary_features_cols
from cmmrt.rt.data import is_non_retained


class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, storage, study_prefix, desc_cols, fgp_cols, n_trials, search_cv,
                 p=0.9, cor_th=0.9, k='all'):
        self.storage = storage
        self.study_prefix = study_prefix
        self.desc_cols = desc_cols
        self.fgp_cols = fgp_cols
        self.p = p
        self.cor_th = cor_th
        self.k = k
        self.n_trials = n_trials
        self.search_cv = search_cv

    def _init_hidden_models(self):
        self._desc_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('imputation', SimpleImputer(missing_values=np.nan, strategy='median', add_indicator=True)),
            ('var_threshold', VarianceThreshold()),
            ('cor_selector', CorThreshold(threshold=self.cor_th)),
            ('f_selector', SelectKBest(score_func=f_regression, k=self.k))
        ])
        self._fgp_vs = VarianceThreshold(threshold=self.p * (1 - self.p))
        self._clf = create_clf()

    def fit(self, X, y=None):
        self._init_hidden_models()
        X_desc = X[:, self.desc_cols]
        X_fgp = X[:, self.fgp_cols]

        self._desc_pipeline.fit(X_desc, y)
        X_fgp_proc = self._fgp_vs.fit_transform(X_fgp)
        self._clf = train_clf(self._clf, X_fgp_proc, is_non_retained(y), self.n_trials, self.search_cv,
                              storage=self.storage, study_prefix=self.study_prefix)
        return self

    def transform(self, X, y=None):
        X_desc = X[:, self.desc_cols]
        X_fgp = X[:, self.fgp_cols]
        X_desc_proc = self._desc_pipeline.transform(X_desc)
        X_fgp_proc = self._fgp_vs.transform(X_fgp)
        prob_predictions = self._clf.predict_proba(X_fgp_proc)[:, 1:].astype('float32')
        new_X = np.concatenate([X_desc_proc, X_fgp, prob_predictions], axis=1)
        # Annotate which columns are related to descriptors an fingerprints after transformation. Also, annotate which
        # columns can be considered binary
        self.transformed_desc_cols = np.concatenate([
            np.arange(X_desc_proc.shape[1], dtype='int'),
            np.array([new_X.shape[1] - 1], dtype='int')
        ], axis=0)
        self.transformed_fgp_cols = np.arange(X_desc_proc.shape[1], new_X.shape[1], dtype='int')
        self.transformed_binary_cols = binary_features_cols(new_X)
        return new_X

    def describe_transformed_features(self):
        return {
            'n_descriptors': len(self.transformed_desc_cols),
            'n_fgp': len(self.transformed_fgp_cols),
            'binary_cols': self.transformed_binary_cols,
            'desc_cols': self.transformed_desc_cols,
            'fgp_cols': self.transformed_fgp_cols
        }

    def transformed_descriptors(self, X):
        return X[:, self.transformed_desc_cols]

    def transformed_fingerprints(self, X):
        return X[:, self.transformed_fgp_cols]

    def _predict_clf_proba(self, X, y=None):
        X_fgp = X[:, self.fgp_cols]
        X_fgp_proc = self._fgp_vs.transform(X_fgp)
        return self._clf.predict_proba(X_fgp_proc).astype('float32')
