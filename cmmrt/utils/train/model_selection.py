import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def stratify_y(y, n_strats=6):
    ps = np.linspace(0, 1, n_strats)
    # make sure last is 1, to avoid rounding issues
    ps[-1] = 1
    quantiles = np.quantile(y, ps)
    cuts = pd.cut(y, quantiles, include_lowest=True)
    codes, _ = cuts.factorize()
    return codes


def stratified_train_test_split(X, y, *, test_size, n_strats=6):
    return train_test_split(X, y, test_size=test_size, stratify=stratify_y(y, n_strats))
