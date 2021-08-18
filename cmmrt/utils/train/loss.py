from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error


def truncated_medae(y_true, y_pred):
    n = len(y_pred)
    return median_absolute_error(y_true[:n], y_pred)


def truncated_rmse(y_true, y_pred):
    n = len(y_pred)
    return mean_squared_error(y_true[:n], y_pred, squared=False)


truncated_medae_scorer = make_scorer(truncated_medae, greater_is_better=False)
truncated_rmse_scorer = make_scorer(truncated_rmse, greater_is_better=False)
