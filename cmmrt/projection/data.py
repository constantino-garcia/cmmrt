"""Functions to load and process data related to projections between chromatographic methods."""
import os

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from cmmrt.projection.models.preprocessor.rt_transformer import RTTransformer

_CUTOFF = 5


def load_xabier_projections(download_directory="rt_data", min_points=0, remove_non_retained=False):
    """Downloads PredRet dataset used in the paper
    <<Domingo-Almenara, Xavier, et al. "The METLIN small molecule dataset for machine learning-based retention
    time prediction." Nature communications 10.1 (2019): 1-9>>
    for assessing the performance of projection methods.
    """
    filename = os.path.join(download_directory, "projections_database.csv")
    url = "https://drive.google.com/u/0/uc?id=1WwySS_FxcyjBUnqTAMfOpy2H7IQf4Kmc&export=download"
    if not os.path.exists(filename):
        if not os.path.exists(os.path.dirname(filename)):
            os.mkdir(os.path.dirname(filename))
        import gdown
        gdown.download(url, filename)
    dat = pd.read_csv(filename)
    dat = dat[['system', 'rt', 'model_prediction']].rename(columns={'rt': 'rt_exper', 'model_prediction': 'rt_pred'})
    dat.rt_pred = dat.rt_pred / 600
    if remove_non_retained:
        # Conservative cutoff. 600
        # Use 3000 to use same definition of 'non-retained' as when training
        dat = dat[dat.rt_pred > _CUTOFF]

    count_by_system = dat['system'].value_counts()
    # This filtering is aligned with the RANGE_TRAINING_POINTS and the fact that we don't want
    # to use more than half data as "annotated samples"
    large_systems = count_by_system[count_by_system > min_points].index
    dat = dat[dat['system'].isin(large_systems)]
    systems = np.unique(dat['system'].values)
    x_scaler, y_scaler = _create_scalers(dat)

    return dat, systems, x_scaler, y_scaler


def _bootstrap_cmm_predret():
    """Create predret by ensembling several datasets. This is only kept for reference on the
    creation process. Use load cmm_predret to load the final dataset."""
    predret = pd.read_csv("rt_data/predret.csv")
    predret.drop(["Unnamed: 0", "Name"], axis=1, inplace=True)
    predret.dropna(inplace=True)
    predret = predret.astype({'Pubchem': int})
    predret.head()
    predret.rename({'RT': 'rt_exper', 'System': 'system'}, axis=1, inplace=True)

    predictions = load_cmm_predictions()
    cmm_predret = predret.merge(predictions, on="Pubchem")
    cmm_predret.to_csv("rt_data/cmm_predret.csv", index=False)


def load_cmm_predictions():
    # TODO
    # The following predictions were created using predict_with_dnn (see DNN branch) on CMM (CMM_predictions)
    # and a custom list of missing molecules available in Predret but not in the paper
    predictions = pd.concat([
        # FIXME
        pd.read_csv("/data/code/research/cembio/cmmrt/results/feb_5/CMM_predictions.csv"),
        pd.read_csv("/data/code/research/cembio/cmmrt/results/feb_5/missing_predictions.csv"),
        pd.read_csv("/data/code/research/cembio/cmmrt/results/feb_5/missing_predictions2.csv")
    ])
    predictions.rename({'pid': 'Pubchem', 'prediction': 'rt_pred'}, axis=1, inplace=True)
    predictions = predictions[predictions['Pubchem'] != "\\N"]
    predictions = predictions.astype({'Pubchem': int})
    return predictions.drop_duplicates()


# TODO: avoid code duplication with load_xabier_projections
def load_cmm_projections(download_directory="rt_data", min_points=0, remove_non_retained=False):
    filename = os.path.join(download_directory, "cmm_predret.csv")
    url = "https://drive.google.com/u/0/uc?id=1aiFDbvnwFhsTrW9jAyGK8tgBssn-wUXF&export=download"
    if not os.path.exists(filename):
        if not os.path.exists(os.path.dirname(filename)):
            os.mkdir(os.path.dirname(filename))
        import gdown
        gdown.download(url, filename)
    dat = pd.read_csv(filename)
    dat.rt_pred = dat.rt_pred / 60
    if remove_non_retained:
        dat = dat[dat.rt_pred > _CUTOFF]

    count_by_system = dat['system'].value_counts()
    # This filtering is aligned with the RANGE_TRAINING_POINTS and the fact that we don't want
    # to use more than half data as "annotated samples"
    large_systems = count_by_system[count_by_system > min_points].index
    dat = dat[dat['system'].isin(large_systems)]
    systems = np.unique(dat['system'].values)
    x_scaler, y_scaler = _create_scalers(dat)
    return dat, systems, x_scaler, y_scaler


def _create_scalers(dat):
    x_scaler = RTTransformer('normalization')
    x_scaler.fit(dat['rt_pred'].values.reshape(-1, 1))
    y_scaler = RTTransformer('standardization')
    y_scaler.fit(dat['rt_pred'].values.reshape(-1, 1))
    return x_scaler, y_scaler


# TODO: download if not exist
def load_predret(download_directory="rt_data"):
    """PredRet Database"""
    return pd.read_csv("rt_data/predret.csv").drop("Unnamed: 0", axis=1)


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


def load_predret_cutoffs():
    return pd.DataFrame([
        ("FEM_orbitrap_plasma", 2),
        ("RIKEN", 1),
        ("MTBLS38", 2.5),
        ("LIFE_old", 1),
        ("LIFE_new", 1),
        ("FEM_long", 5),
        ("MTBLS87", 5),
        ("MTBLS36", 2.5),
        ("MTBLS20", 1.5),
        ("FEM_short", 1),
        ("PFR - TK72", 1),
        ("INRA_QTOF", 2),
        ("FEM_orbitrap_urine", 2.5),
        ("Qtof - PFEM", 2),
        ("Waters ACQUITY UPLC with Synapt G1 Q-TOF", 1),
        ("OBSF", 1),
        ("Cao_HILIC", 5),
        ("IPB_Halle", 2.25),
        ("UFZ_Phenomenex", 5),
        ("UniToyama_Atlantis", 5),
        ("MTBLS39", 1),
        ("FEM_lipids", 5),
        ("Eawag_XBridgeC18", 3),
        ("MPI_Symmetry", 2),
        ("MTBLS4", 2),
        ("X1290SQ", 10),
        ("MTBLS17", 1),
        ("MTBLS19", 2),
        ("MTBLS52", 1),
        ("1290SQ", 0)
    ], columns=['system', 'cutoff'])


def load_predicted_predret(dataset, download_directory, remove_non_retained):
    print(f"Loading Predret with {dataset} predictions...")
    if dataset == "xabier":
        load_function = load_xabier_projections
    elif dataset == "cmm":
        load_function = load_cmm_projections
    else:
        raise ValueError("Unknown dataset")
    data, systems, x_scaler, y_scaler = load_function(download_directory, remove_non_retained=remove_non_retained)
    if remove_non_retained:
        cutoffs = load_predret_cutoffs()
        data = pd.merge(data, cutoffs, on='system')
        data = data[data.rt_exper > data.cutoff]
    return data, systems, x_scaler, y_scaler


def get_representatives(x, y, test_size, n_quantiles=10, random_state_generator=None,
                        return_complement=False):
    """Get representative samples of the (x, y) dataset by selecting points that 'cover' the x range."""
    x_ndim = x.ndim
    y_ndim = y.ndim
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    # First sample() performs stratified sampling, second sample() limits the number
    # of samples to test_size
    # groups = pd.qcut(x.flatten(), n_quantiles)
    kmeans = KMeans(n_quantiles, random_state=0).fit(x.reshape(-1, 1))
    groups = kmeans.labels_

    df = pd.DataFrame({'x': x.flatten(), 'y': y.flatten(), 'group': groups})
    random_state = next(random_state_generator) if random_state_generator is not None else None
    representatives = df.groupby(df.group).sample(1, random_state=random_state)
    while len(representatives) < test_size:
        n_to_go = test_size - len(representatives)
        not_sampled = df[~df.isin(representatives).all(1)]
        random_state = next(random_state_generator) if random_state_generator is not None else None
        new_samples = not_sampled.groupby(not_sampled.group).sample(1, random_state=random_state)
        if len(new_samples) > n_to_go:
            random_state = next(random_state_generator) if random_state_generator is not None else None
            new_samples = new_samples.sample(n_to_go, random_state=random_state)
        representatives = pd.concat([representatives, new_samples], axis=0)

    if return_complement:
        not_in_representatives = df.merge(representatives, indicator=True, how='left').loc[
            lambda x: x['_merge'] != 'both']
        if len(not_in_representatives) + len(representatives) != len(df):
            print(f'----------> {len(not_in_representatives)} + {len(representatives)} != {len(df)}')
        xnr = not_in_representatives.x.values
        ynr = not_in_representatives.y.values
        if x_ndim == 2:
            xnr = xnr.reshape(-1, 1)
        if y_ndim == 2:
            ynr = ynr.reshape(-1, 1)

    xr = representatives.x.values
    yr = representatives.y.values
    if x_ndim == 2:
        xr = xr.reshape(-1, 1)
    if y_ndim == 2:
        yr = yr.reshape(-1, 1)

    if return_complement:
        return xr, yr, xnr, ynr
    else:
        return xr, yr
