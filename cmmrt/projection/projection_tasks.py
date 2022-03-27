import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class ProjectionsTasks(Dataset):
    """Meta-learning dataset where each task consists of projecting retention times from the SMRT dataset
    to the retention times as measured in a different chromatography system."""

    def __init__(self, projections_dat, direction="p2e", p_support_range=(1, 1), min_n=20, x_scaler=None,
                 y_scaler=None):
        """
        :param projections_dat: pandas dataframe with information of the retention times predicted by a machine
        learning model (column 'rt_pred') and the retention times measured ('rt_exper') in different
        chromatography systems ('system').
        :param direction: 'p2e' (predicted to experimental) or 'e2p' (experimental 2 predicted)
        :param p_support_range: proportion of the systems' data used for creating a projection task specified
        as a tuple (min_p, max_p). That is, to create a projection tasks for a given system, a random proportion
        p from the range (min_p, max_p) is drawn. Then, a random subset of the systems' data is selected to create
        a projection task.
        :param min_n: minimum number of samples for a system to be considered for creating a projection task.
        :param x_scaler: Scikit-learn transformer or None. If provided, the scaler is applied to the retention times in the x-axis.
        :param y_scaler: Scikit-learn transformer or None. If provided, the scaler is applied to the retention times in the y-axis.
        """
        assert direction in ["p2e", "e2p"], "Invalid direction. Should be one of 'p2e' or 'e2p'"
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
        self.direction = direction
        self.p_support_range = p_support_range
        self.scaler_x = x_scaler
        self.scaler_y = y_scaler

    def __len__(self):
        return len(self.systems)

    def __getitem__(self, idx):
        system_data = self.projections_dat.loc[self.projections_dat.system == self.systems[idx], :]
        if self.direction == "p2e":
            x = system_data['rt_pred'].values.reshape(-1, 1)
            y = system_data['rt_exper'].values
        else:
            x = system_data['rt_exper'].values.reshape(-1, 1)
            y = system_data['rt_pred'].values

        if self.p_support_range[0] == self.p_support_range[1]:
            n_support = int(self.p_support_range[0] * len(y))
        else:
            n_support_range = (np.array(self.p_support_range) * len(y)).astype('int')
            n_support = int(np.random.randint(*n_support_range))

        if n_support < len(y):
            _, x_support, _, y_support = train_test_split(x, y, test_size=n_support)
        else:
            x_support, y_support = x, y

        if self.scaler_x:
            x_support = self.scaler_x.transform(x_support)
        if self.scaler_y:
            y_support = self.scaler_y.transform(y_support.reshape(-1, 1)).flatten()
        return (
            torch.from_numpy(x_support).float(),
            torch.from_numpy(y_support).float(),
            self.systems[idx]
        )

    def inverse_transform(self, x, y):
        x = self.scaler_x.inverse_transform(x.reshape(-1, 1)).reshape(x.shape)
        y = self.scaler_y.inverse_transform(y.reshape(-1, 1)).reshape(y.shape)
        return x, y

    def inverse_ci(self, mean, var):
        return self.scaler_y.inverse_ci(mean, var)
