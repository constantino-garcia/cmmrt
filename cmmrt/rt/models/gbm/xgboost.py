import copy
import inspect

import optuna
from optuna.trial import TrialState
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor, XGBClassifier

from cmmrt.rt.models.base.PipelineWrapper import RTRegressor

_ACCURACY = False


class WeightedXGBClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=None, max_depth=None, learning_rate=None, verbosity=None, booster=None, n_jobs=None,
                 gamma=None, min_child_weight=None, subsample=None, colsample_bytree=None, colsample_bynode=None,
                 colsample_bylevel=None, reg_alpha=None, reg_lambda=None, tree_method=None):
        super().__init__()
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)
        print('Using a WeightedXGBClassifier!')

    def _get_xgb_params(self):
        params = copy.deepcopy(self.get_params())
        return params

    def fit(self, X, y):
        params = self._get_xgb_params()
        self._model = XGBClassifier(**params, use_label_encoder=False)
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        return self._model.predict_proba(X)


def _get_study_params(study_prefix=''):
    # FIXME: should be tunable by the user!
    _STORAGE = 'sqlite:///retained.db'
    name = 'retained_non-retained' if study_prefix == '' else study_prefix + '-retained_non-retained'
    return name, _STORAGE


def create_clf(**params):
    return WeightedXGBClassifier(**params)


def create_clf_objective(X, y, cv):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int('n_estimators', 0, 1000),
            'max_depth': trial.suggest_int('max_depth', 2, 25),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.005, 0.5),
            'verbosity': 0,
            # Not using Dart at the moment
            'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear']),
            'n_jobs': -1,
            'gamma': trial.suggest_int('gamma', 0, 5),
            'min_child_weight': trial.suggest_int('min_child_weight', 0, 5),
            'subsample': trial.suggest_discrete_uniform('subsample', 0.5, 1, 0.05),
            'colsample_bytree': trial.suggest_discrete_uniform('colsample_bytree', 0.1, 1, 0.01),
            'colsample_bynode': trial.suggest_discrete_uniform('colsample_bynode', 0.1, 1, 0.01),
            'colsample_bylevel': trial.suggest_discrete_uniform('colsample_bylevel', 0.1, 1, 0.01),
            'reg_alpha': trial.suggest_int('reg_alpha', 0, 5),
            'reg_lambda': trial.suggest_int('reg_lambda', 0, 5),
            'tree_method': trial.suggest_categorical('tree_method', ['exact', 'approx', 'hist'])
        }
        clf = create_clf(**params)
        # Do not use parallelization here since it is already used within the booster
        if _ACCURACY:
            return cross_val_score(clf, X, y, cv=cv, n_jobs=1, scoring='accuracy').mean()
        else:
            return cross_val_score(clf, X, y, cv=cv, n_jobs=1, scoring='f1').mean()
    return objective


def load_best_clf_params(storage, study_prefix=''):
    study_name = build_study_name(study_prefix)
    study = optuna.load_study(
        study_name=study_name,
        storage=storage
    )
    return study.best_params


def train_clf(clf, X, y, n_trials, search_cv, storage, study_prefix='', keep_going=False):
    study_name = build_study_name(study_prefix)
    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',
        storage=storage,
        load_if_exists=True
    )
    if not keep_going:
        trials = [trial for trial in study.get_trials() if trial.state in [TrialState.COMPLETE, TrialState.PRUNED]]
        n_trials = n_trials - len(trials)
    if n_trials > 0:
        objective = create_clf_objective(X, y, search_cv)
        study.optimize(objective, n_trials=n_trials)

    clf.set_params(**load_best_clf_params(storage, study_prefix))
    return clf.fit(X, y)


def build_study_name(study_prefix):
    if _ACCURACY:
        print('Using accuracy approach!')
        study_name = 'retained_non-retained' if study_prefix == '' else study_prefix + '-retained_non-retained'
    else:
        study_name = 'retained_non-retained-f1' if study_prefix == '' else study_prefix + '-retained_non-retained-f1'
    return study_name


class SelectiveXGBRegressor(RTRegressor):
    def __init__(self, n_estimators=500, max_depth=3, learning_rate=0.1, booster='gblinear',
                 gamma=1.8, min_child_weight=0.18, subsample=0.95, reg_alpha=0.07, reg_lambda=3.6,
                 colsample_bytree=0.45, colsample_bylevel=0.48, colsample_bynode=0.93,
                 tree_method='hist', verbosity=1, n_jobs=-1,
                 use_col_indices='all', binary_col_indices=None, var_p=0, transform_output=False
                 ):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)

    def _init_regressor(self):
        return XGBRegressor(**self._rt_regressor_params())
