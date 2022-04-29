"""Functions to load and process data related to projections between chromatographic methods."""
import os

import importlib_resources
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans

from cmmrt.projection.models.preprocessor.rt_transformer import RTTransformer

_CUTOFF = 5


def _load_predret_from_url(filename, url, download_directory, rt_scale, min_points, remove_non_retained,
                           renaming_dict=None):
    filename = os.path.join(download_directory, filename)
    if not os.path.exists(filename):
        if not os.path.exists(os.path.dirname(filename)):
            os.mkdir(os.path.dirname(filename))
        import gdown
        gdown.download(url, filename)
    dat = pd.read_csv(filename)
    if renaming_dict is not None:
        dat = dat[['pubchem', 'system', 'rt', 'model_prediction']].rename(
            columns=renaming_dict
        )
    dat.rt_pred = dat.rt_pred / rt_scale
    if remove_non_retained:
        dat = dat[dat.rt_pred > _CUTOFF]

    count_by_system = dat['system'].value_counts()
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


def _load_predret_with_xabier_predictions(download_directory="rt_data", min_points=0, remove_non_retained=False):
    """Downloads PredRet dataset used in the paper
    <<Domingo-Almenara, Xavier, et al. "The METLIN small molecule dataset for machine learning-based retention
    time prediction." Nature communications 10.1 (2019): 1-9>>
    for assessing the performance of projection methods.
    """
    return _load_predret_from_url(
        filename="projections_database.csv",
        url="https://drive.google.com/u/0/uc?id=1WwySS_FxcyjBUnqTAMfOpy2H7IQf4Kmc&export=download",
        download_directory=download_directory,
        rt_scale=600,
        min_points=min_points,
        remove_non_retained=remove_non_retained,
        renaming_dict={'pubchem': 'Pubchem', 'rt': 'rt_exper', 'model_prediction': 'rt_pred'}
    )


# def _bootstrap_cmm_predret():
#     """Create predret by ensembling several datasets. This is only kept for reference on the
#     creation process. Use load cmm_predret to load the final dataset."""
#     predret = pd.read_csv("rt_data/predret.csv")
#     predret.drop(["Unnamed: 0", "Name"], axis=1, inplace=True)
#     predret.dropna(inplace=True)
#     predret = predret.astype({'Pubchem': int})
#     predret.head()
#     predret.rename({'RT': 'rt_exper', 'System': 'system'}, axis=1, inplace=True)
#
#     predictions = load_cmm_predictions()
#     cmm_predret = predret.merge(predictions, on="Pubchem")
#     cmm_predret.to_csv("rt_data/cmm_predret.csv", index=False)

def _load_predret_with_cmm_predictions(download_directory="rt_data", min_points=0, remove_non_retained=False):
    return _load_predret_from_url(
        filename="cmm_predret.csv",
        url="https://drive.google.com/u/0/uc?id=1aiFDbvnwFhsTrW9jAyGK8tgBssn-wUXF&export=download",
        download_directory=download_directory,
        rt_scale=60,
        min_points=min_points,
        remove_non_retained=remove_non_retained
    )


def load_predret():
    """PredRet Database"""
    path = importlib_resources.files("cmmrt.data").joinpath("predret.csv")
    return pd.read_csv(path).drop("Unnamed: 0", axis=1)


def load_predret_with_predictions(dataset, download_directory, remove_non_retained):
    print(f"Loading Predret with {dataset} predictions...")
    if dataset == "xabier":
        load_function = _load_predret_with_xabier_predictions
    elif dataset == "cmm":
        load_function = _load_predret_with_cmm_predictions
    else:
        raise ValueError("Unknown dataset")
    data, systems, x_scaler, y_scaler = load_function(download_directory, remove_non_retained=remove_non_retained)
    # TODO: move to _load_predret_from_url
    if remove_non_retained:
        cutoffs = _load_predret_cutoffs()
        data = pd.merge(data, cutoffs, on='system')
        data = data[data.rt_exper > data.cutoff]
    return data, systems, x_scaler, y_scaler


def _load_predret_cutoffs():
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
