import os

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from torch.utils.data import Dataset


class Detrender(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        self._lm = LinearRegression().fit(X, y)
        return self

    def transform(self, y):
        return (y - self._lm.intercept_) / self._lm.coef_

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(y)

    def inverse_transform(self, y):
        return y * self._lm.coef_ + self._lm.intercept_

    def inverse_var_transform(self, var):
        return var * self._lm.coef_ ** 2


class RTTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self._scaler = RobustScaler()
        self._scaler.fit(np.log(X))
        return self

    def transform(self, X):
        try:
            return self._scaler.transform(np.log(1 + X))
        except Exception as e:
            print(e)

    def inverse_transform(self, X):
        return np.exp(self._scaler.inverse_transform(X)) - 1

    def inverse_ci(self, mean, var, z=1.96):
        new_mean = self._scaler.inverse_transform(mean.reshape(-1, 1)).flatten()
        new_var = var * self._scaler.scale_ ** 2
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


def load_xabier_projections(download_directory="rt_data", min_points=0, remove_non_retained=False):
    filename = os.path.join(download_directory, "projections_database.csv")
    url = "https://drive.google.com/u/0/uc?id=1WwySS_FxcyjBUnqTAMfOpy2H7IQf4Kmc&export=download"
    if not os.path.exists(filename):
        if not os.path.exists(os.path.dirname(filename)):
            os.mkdir(os.path.dirname(filename))
        import gdown
        gdown.download(url, filename)
    dat = pd.read_csv(filename)
    dat = dat[['system', 'rt', 'model_prediction']].rename(columns={'rt': 'rt_exper', 'model_prediction': 'rt_pred'})
    if remove_non_retained:
        # Conservative cutoff. Use 3000 to use same definition of 'non-retained' as when training
        dat = dat[dat.rt_pred > 600]

    dat.rt_pred = dat.rt_pred / 600
    count_by_system = dat['system'].value_counts()
    # This filtering is aligned with the RANGE_TRAINING_POINTS and the fact that we don't want
    # to use more than half data as "annotated samples"
    large_systems = count_by_system[count_by_system > min_points].index
    dat = dat[dat['system'].isin(large_systems)]
    systems = np.unique(dat['system'].values)
    scaler = RTTransformer()
    scaler.fit(dat['rt_pred'].values.reshape(-1, 1))

    return dat, systems, scaler


class ProjectionsTasks(Dataset):
    def __init__(self, projections_dat, p_support_range, min_n=20, scaler=None):
        assert len(p_support_range) == 2, 'p_support_range should be a duple (min_p, max_p)'
        assert 0 <= p_support_range[0] <= 1, 'invalid p_support_range'
        assert 0 <= p_support_range[1] <= 1, 'invalid p_support_range'
        system_counts = projections_dat['system'].value_counts()
        self.systems = (
            np.array(system_counts[system_counts >= min_n].index)
        )
        self.projections_dat = projections_dat[projections_dat.system.isin(self.systems)]
        self.projections_dat = self.projections_dat.astype(
            {'rt_exper': 'float32', 'rt_pred': 'float32'}
        )
        self.p_support_range = p_support_range
        self.scaler = scaler

    def __len__(self):
        return len(self.systems)

    def __getitem__(self, idx):
        system_data = self.projections_dat.loc[self.projections_dat.system == self.systems[idx], :]
        x = system_data['rt_pred'].values.reshape(-1, 1)
        y = system_data['rt_exper'].values
        if self.p_support_range[0] == self.p_support_range[1]:
            n_support = int(self.p_support_range[0] * len(y))
        else:
            n_support_range = (np.array(self.p_support_range) * len(y)).astype('int')
            n_support = int(np.random.randint(*n_support_range))

        if n_support < len(y):
            _, x_support, _, y_support = train_test_split(x, y, test_size=n_support)
        else:
            x_support, y_support = x, y

        if self.scaler:
            x_support = self.scaler.transform(x_support)
            y_support = self.scaler.transform(y_support.reshape(-1, 1)).flatten()
        return x_support, y_support
