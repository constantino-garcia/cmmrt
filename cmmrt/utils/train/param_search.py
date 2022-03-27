from functools import singledispatch

import lightgbm as lgb
import numpy as np
import optuna
from gpytorch.utils.errors import NotPSDError
from lightgbm import LGBMRegressor
from optuna.integration import LightGBMTunerCV
from optuna.trial import TrialState
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from xgboost import XGBClassifier
from xgboost import XGBRegressor

from cmmrt.rt.models.gbm.xgboost import SelectiveXGBRegressor
from cmmrt.rt.models.ensemble.Blender import Blender
from cmmrt.rt.models.gp.DKL import SkDKL
from cmmrt.rt.models.nn.SkDnn import SkDnn
from cmmrt.utils.train.loss import truncated_medae_scorer


@singledispatch
def suggest_params(estimator, trial):
    raise NotImplementedError


@suggest_params.register
def _(estimator: SelectiveXGBRegressor, trial):
    return _suggest_xgboost(trial)


@suggest_params.register
def _(estimator: XGBRegressor, trial):
    return _suggest_xgboost(trial)


@suggest_params.register
def _(estimator: XGBClassifier, trial):
    return _suggest_xgboost(trial)


def _suggest_xgboost(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'max_depth': trial.suggest_int('max_depth', 1, 31),  # max_depth cannont be greater than 31 to use gpu_hist
        'learning_rate': trial.suggest_uniform('learning_rate', 1e-4, 2e-1),
        'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear']),
        'gamma': trial.suggest_uniform('gamma', 0, 2),
        'min_child_weight': trial.suggest_loguniform('min_child_weight', 0.001, 10),
        'subsample': trial.suggest_uniform('subsample', 0.4, 1.0),
        'reg_alpha': trial.suggest_uniform('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_uniform('reg_lambda', 0, 5),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.4, 1.0),
        'colsample_bylevel': trial.suggest_uniform('colsample_bylevel', 0.4, 1.0),
        'colsample_bynode': trial.suggest_uniform('colsample_bynode', 0.4, 1.0),
        'tree_method': trial.suggest_categorical('tree_method', ['exact', 'approx', 'hist']),  # gpu_hist may be added
        'verbosity': 1,
        'var_p': trial.suggest_uniform('var_p', 0.9, 1.0)
    }
    params['n_jobs'] = 1 if params['tree_method'] == 'gpu_hist' else -1
    return params


@suggest_params.register
def _(estimator: SkDKL, trial):
    scheduler_patience = trial.suggest_int('scheduler_patience', 5, 20)
    params = {
        'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'mixture']),
        'hidden_1': trial.suggest_categorical('hidden_1', [512, 1024, 2048, 4096]),
        'hidden_2': trial.suggest_categorical('hidden_2', [64, 128, 256, 512, 1024]),
        'dropout': trial.suggest_uniform('dropout', 0, 0.7),
        'use_bn_out': trial.suggest_categorical('use_bn_out', [True, False]),
        'lr': trial.suggest_uniform('lr', 1e-4, 1e-1),
        'scheduler_patience': scheduler_patience,
        'early_stopping': trial.suggest_int('early_stopping',
                                            scheduler_patience + 2, 5 * scheduler_patience + 2),
        'var_p': trial.suggest_uniform('var_p', 0.9, 1)
    }
    return params


@suggest_params.register
def _(estimator: LGBMRegressor, trial):
    params = {
        'objective': 'regression',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    }

    return params


@suggest_params.register
def _(estimator: SkDnn, trial):
    h1 = trial.suggest_categorical('hidden_1', [512, 1024, 1512, 2048, 4096])
    T0 = trial.suggest_int('T0', 10, 100)
    params = {
        'hidden_1': h1,
        'hidden_2': trial.suggest_int('hidden_2', 32, 512),
        'dropout_1': trial.suggest_uniform('dropout_1', 0.3, 0.7),
        'dropout_2': trial.suggest_uniform('dropout_2', 0.0, 0.2),
        'activation': trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'gelu', 'swish']),
        'lr': trial.suggest_uniform('lr', 1e-4, 1e-3),
        'T0': T0,
        'annealing_rounds': trial.suggest_int('annealing_rounds', 2, 5),
        'swa_epochs': trial.suggest_int('swa_epochs', 5, T0),
        'var_p': trial.suggest_uniform('var_p', 0.9, 1.0)
    }
    return params


@suggest_params.register
def _(estimator: RandomForestRegressor, trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 10, 500),
        'max_depth': trial.suggest_int('max_depth', 1, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 150),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 500),
        'max_features': trial.suggest_categorical('max_features', ["auto", "sqrt", "log2"]),
        'n_jobs': -1
    }


def create_objective(estimator, X, y, cv):
    def estimator_factory():
        return clone(estimator)

    def objective(trial):
        estimator = estimator_factory()
        params = suggest_params(estimator, trial)
        estimator.set_params(**params)
        scoring = truncated_medae_scorer
        try:
            score = cross_val_score_with_pruning(estimator, X, y, cv=cv, scoring=scoring, trial=trial)
        except NotPSDError:
            print('NotPSDError while cross-validating')
            score = -np.inf
        return score

    return objective


def cross_val_score_with_pruning(estimator, X, y, cv, scoring, trial):
    cross_val_scores = []
    for step, (train_index, test_index) in enumerate(cv.split(X, y)):
        est = clone(estimator)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        est.fit(X_train, y_train)
        cross_val_scores.append(scoring(est, X_test, y_test))
        intermediate_value = np.mean(cross_val_scores)
        trial.report(intermediate_value, step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return np.mean(cross_val_scores)


@singledispatch
def final_estimator_study_name(estimator):
    raise NotImplementedError


@final_estimator_study_name.register
def _(estimator: RandomForestRegressor):
    return 'rf_final_est'


@final_estimator_study_name.register
def _(estimator: ElasticNet):
    return 'eln_final_est'


@singledispatch
def param_search(estimator, X, y, cv, study, n_trials, keep_going=False):
    objective = create_objective(estimator, X, y, cv)
    trials = [trial for trial in study.get_trials() if trial.state in [TrialState.COMPLETE, TrialState.PRUNED]]
    if not keep_going:
        n_trials = n_trials - len(trials)
    if n_trials > 0:
        print(f"Starting {n_trials} trials")
        study.optimize(objective, n_trials=n_trials)

    return load_best_params(estimator, study)


@param_search.register
def _(estimator: LGBMRegressor, X, y, cv, study, n_trials, keep_going=False):
    dtrain = lgb.Dataset(X, label=y)
    trials = [trial for trial in study.get_trials() if trial.state in [TrialState.COMPLETE, TrialState.PRUNED]]
    # LightGBMTunerCV always runs 68 trials
    if len(trials) != 68:
        tuner = _create_lgbm_tuner(dtrain, study, cv)
        tuner.run()
    return load_best_params(estimator, study)


@param_search.register
def _(estimator: Blender, X, y, cv, study, n_trials, keep_going=False):
    # For the blender, the study is expected to consist of a duple: (storage, study_prefix)
    storage, study_prefix = study
    X_train, X_test, y_train, y_test = estimator._blending_split(X, y)

    models_with_studies = []
    for model_name, model in estimator.estimators:
        if 'cb' in model_name:
            print(f'Skipping optimization for {model_name}')
            models_with_studies.append((model_name, model, None))
            continue
        else:
            model = clone(model)
            study = create_study(model_name, study_prefix, storage)
            models_with_studies.append((model_name, model, study))
            _ = param_search(model, X_train, y_train, cv,
                             study, n_trials, keep_going=keep_going)

    estimator.estimators = [
        (n, set_best_params(clone(model), study)) for n, model, study in models_with_studies
    ]

    # Train with best parameters and predict to create the dataset for the blender
    blended_X = []
    fitted_estimators = []
    for model_name, model, study in models_with_studies:
        # train_with_best clones first
        if study is None:
            model = model.fit(X_train, y_train)
        else:
            model = train_with_best_params(model, X_train, y_train, study)
        fitted_estimators.append((model_name, model))
        blended_X.append(model.predict(X_test).reshape(-1, 1))

    blended_dataset = {
        'X': np.concatenate(blended_X, axis=1),
        'y': y_test
    }
    final_estimator = clone(estimator.final_estimator)
    study = create_study(final_estimator_study_name(final_estimator), study_prefix, storage)

    _ = param_search(final_estimator, blended_dataset['X'], blended_dataset['y'],
                     cv, study, n_trials, keep_going=False)
    estimator.final_estimator = set_best_params(clone(estimator.final_estimator), study)
    return estimator


def create_study(model_name, study_prefix, storage):
    return optuna.create_study(
        study_name=f'{study_prefix}-{model_name}',
        direction='minimize' if model_name == 'lgb' else 'maximize',
        storage=storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner()
    )


def _create_lgbm_tuner(dtrain, study, cv):
    params = {
        "objective": "regression",
        "metric": "l1",
        "verbosity": -1,
        "boosting_type": "gbdt",
    }
    tuner = LightGBMTunerCV(
        params, dtrain, verbose_eval=True, early_stopping_rounds=100, folds=cv, study=study
    )
    return tuner


@singledispatch
def set_best_params(estimator, study):
    if study is not None:
        best_params = load_best_params(estimator, study)
        estimator.set_params(**best_params)
    return estimator


@set_best_params.register
def _(estimator: Blender, study):
    # Again (see param_search), for the blender the study is expected to consist of a duple (storage, study_prefix)
    storage, study_prefix = study
    estimator.estimators = [
        (n, set_best_params(clone(model), create_study(n, study_prefix, storage))) for (n, model) in estimator.estimators
    ]
    estimator.final_estimator = set_best_params(
        clone(estimator.final_estimator),
        create_study(final_estimator_study_name(estimator.final_estimator), study_prefix, storage)
    )
    return estimator


@singledispatch
def train_with_best_params(estimator, X, y, study):
    estimator = clone(estimator)
    best_params = load_best_params(estimator, study)
    estimator.set_params(**best_params)
    return estimator.fit(X, y)


@singledispatch
def load_best_params(estimator, study):
    try:
        return study.best_params
    except Exception as e:
        print(f'Study for {type(estimator)} does not exist')
        raise e


@load_best_params.register
def _(estimator: LGBMRegressor, study):
    tuner = _create_lgbm_tuner(None, study, None)
    return tuner.best_params
