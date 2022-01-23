"""Functions to load the SMRT dataset and fingerprints/descriptors computed with Alvadesc"""
import bz2
import copy
import os
import pickle
import warnings
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from cmmrt.projection.data import load_predret


class FeaturizedRtData(ABC):
    """Class representing featurized molecules with either fingerprints or descriptors or both, and their associated
    retention times.

    Attributes:
        X: 2D numpy array containing the Alvadesc's descriptors and fingerprints of each molecule in the dataset.
        fingerprints: 2D numpy array containing the Alvadesc's fingerprints of each molecule in the dataset.
        descriptors: 2D numpy array containing the Alvadesc's descriptors of each molecule in the dataset.
        fgp_cols: 1D numpy array containing the indices of the X array that correspond with fingerprints.
        desc_cols: 1D numpy array containing the indices of the X array that correspond with descriptors.
        y: 1D numpy array containing the retention times of each molecule in the dataset.
    """
    def __init__(self, filename, download_directory):
        """Initialize the class.
        :param download_directory: directory where the featurized molecules with their retention times are downloaded.
        """
        filename = os.path.join(download_directory, filename)
        if not os.path.exists(filename):
            data = self._create_dict_dataset(download_directory)
            # check correctness of the dictionary
            for name in ['X', 'y', 'desc_cols', 'fgp_cols']:
                assert name in data.keys(), "Missing key {} in data".format(name)
            for key in data:
                setattr(self, key, data[key])
            print('saving')
            with bz2.BZ2File(filename, "wb") as f:
                pickle.dump([self.X, self.y, self.desc_cols, self.fgp_cols], f)
        else:
            with bz2.BZ2File(filename, "rb") as f:
                self.X, self.y, self.desc_cols, self.fgp_cols = pickle.load(f)

    @abstractmethod
    def _create_dict_dataset(self, download_directory):
        """Returns a dictionary with X, y, desc_cols, fgp_cols"""
        pass

    @property
    def fingerprints(self):
        """Returns Alvadesc's fingerprints of the dataset.

        :return: Alvadesc's fingerprints of the dataset as a numpy array or None if the dataset does not contain
        fingerprints.
        """
        if self.fgp_cols is not None and len(self.fgp_cols) > 0:
            return self.X[:, self.fgp_cols]
        else:
            warnings.warn("Dataset does not contain fingerprints")
            return None

    @property
    def descriptors(self):
        """Returns Alvadesc's descriptors of the dataset.

        :return: Alvadesc's descriptors of the dataset as a numpy array or None if the dataset does not contain
        descriptors.
        """
        if self.desc_cols is not None and len(self.desc_cols) > 0:
            return self.X[:, self.desc_cols]
        else:
            warnings.warn("Dataset does not contain descriptors")
            return None

    def __getitem__(self, item):
        self_copy = copy.deepcopy(self)
        self_copy.X = self_copy.X[item, :]
        self_copy.y = self_copy.y[item]
        return self_copy


class AlvadescDataset(FeaturizedRtData):
    """Class to load the SMRT dataset where each molecule has been featurized with both fingerprints and descriptors
     using Alvadesc."""

    def __init__(self, download_directory="rt_data"):
        """Initialize the class.
        :param download_directory: directory where the featurized molecules with their retention times are downloaded.
        """
        super().__init__("alvadesc.pklz", download_directory)

    def _create_dict_dataset(self, download_directory):
        common_cols = ['pid', 'rt']
        print('reading fingerprints...')
        fgp = load_alvadesc_fingerprints(download_directory=download_directory, split_as_np=False)
        print('reading descriptors...')
        descriptors = load_alvadesc_descriptors(download_directory=download_directory, split_as_np=False)
        print('merging')
        descriptors = descriptors.drop_duplicates()
        descriptors_fgp = pd.merge(descriptors, fgp, on=common_cols)

        def get_feature_names(x):
            return x.drop(common_cols, axis=1).columns

        X_desc = descriptors_fgp[get_feature_names(descriptors)].values
        X_fgp = descriptors_fgp[get_feature_names(fgp)].values
        X = np.concatenate([X_desc, X_fgp], axis=1)
        return {
            'X': X,
            'y': descriptors_fgp['rt'].values.flatten(),
            'desc_cols': np.arange(X_desc.shape[1], dtype='int'),
            'fgp_cols': np.arange(X_desc.shape[1], X.shape[1], dtype='int')
        }



def load_alvadesc_descriptors(download_directory="rt_data", n=None, split_as_np=True):
    """Downloads the SMRT dataset featurized with Alvadesc's descriptors.

    :param download_directory: directory where the featurized molecules with their retention times are downloaded.
    :param n: if specified, n random samples of the SMRT dataset are returned.
    :param split_as_np: if True, the returned dataframe is split into two numpy arrays (features X, and retention times y).
    If false, the dataset is returned as a pandas dataframe.
    :return: the SMRT dataset featurized with Alvadesc's descriptors.
    """
    filename = "descriptors.pklz"
    filename = os.path.join(download_directory, filename)
    url = "https://drive.google.com/u/0/uc?id=1MMPsk8jghXfzp2DvrRsHsz4iB03shndP&export=download"
    return _load_pickled_data(filename, url, n, split_as_np)


def load_alvadesc_fingerprints(download_directory="rt_data", n=None, split_as_np=True):
    """Downloads the SMRT dataset featurized with Alvadesc's fingerprints.

    :param download_directory: directory where the featurized molecules with their retention times are downloaded.
    :param n: if specified, n random samples of the SMRT dataset are returned.
    :param split_as_np: if True, the returned dataframe is split into two numpy arrays (features X, and retention times y).
    If false, the dataset is returned as a pandas dataframe.
    :return: the SMRT dataset featurized with Alvadesc's fingerprints.
    """
    filename = "fingerprints.pklz"
    filename = os.path.join(download_directory, filename)
    url = "https://drive.google.com/u/0/uc?id=1QQRP559jyjFUQwQVJzZNrEQtlLnIfH8v&export=download"
    return _load_pickled_data(filename, url, n, split_as_np)


def _load_pickled_data(filename, url, n, split_as_np):
    if not os.path.exists(filename):
        if not os.path.exists(os.path.dirname(filename)):
            os.mkdir(os.path.dirname(filename))
        import gdown
        gdown.download(url, filename)
    with bz2.BZ2File(filename, "rb") as f:
        data = pickle.load(f)
    if n:
        data = data.sample(n, axis=1)
    if split_as_np:
        return (
            data.drop(columns=['pid', 'rt'], axis=1).to_numpy(),
            data['rt'].values.reshape(-1, 1)
        )
    else:
        return data


def is_non_retained(rts):
    """Indicates if the given retention times are associated with non-retained molecules.

    :param rts: retention times of the molecules in seconds.
    :return: integer array indicating if the given retention times are associated with non-retained molecules.
    """
    return (rts < 300).astype('int')


def is_binary_feature(x):
    """Indicates if the given feature is binary (0 and 1).

    :param x: feature to be checked.
    :return: Boolean indicating if the given feature is binary.
    """
    ux = np.unique(x)
    if len(ux) == 1:
        return ux == 0 or ux == 1
    if len(ux) == 2:
        return np.all(np.sort(ux) == np.array([0, 1]))
    else:
        return False


def binary_features_cols(X):
    """Get column indices of binary features.

    :param X: numpy array of features.
    :return: numpy array of column indices of binary features.
    """
    return np.where(np.apply_along_axis(is_binary_feature, 0, X))[0]


# TODO: make use of download_directory
def load_cmm_fingerprints(download_directory="rt_data"):
    """Load Alvadesc fingerprints of all molecules included in Ceu Mass Mediator (CMM)"""
    return pd.read_csv("rt_data/CMM_vectorfingerprints.csv")


class PredRetFeaturizedSystem(FeaturizedRtData):
    def __init__(self, system, download_directory="rt_data"):
        """Class to load the retention times from a specific CM of the PredRet database, together with the Alvadesc's fingerprints
        of each molecule.
        """
        self.system = system
        super().__init__(f"{system}_with_alvadesc_features.pklz", download_directory)

    def _create_dict_dataset(self, download_directory):
        predret = load_predret(download_directory)
        cmm_fingerprints = load_cmm_fingerprints(download_directory)
        cmm_fingerprints = cmm_fingerprints[cmm_fingerprints.pid != "\\N"].astype({'pid': 'int'})
        system_data = predret.loc[predret["System"] == self.system]
        fgp_and_rts = pd.merge(
            cmm_fingerprints.drop(['CMM_id'], axis=1),
            system_data.drop(['Name', 'System'], axis=1).rename(columns={'Pubchem': 'pid', 'RT': 'rt'}).astype(
                {'pid': 'int'}),
            on='pid'
        )
        if fgp_and_rts.shape[0] == 0:
            raise ValueError(f"No molecules found for system {self.system}")

        X = fgp_and_rts.drop(['pid', 'rt'], axis=1).values.astype('float32')
        y = fgp_and_rts.rt.values.astype('float32')
        return {
            'X': X,
            'y': y,
            'desc_cols': np.array([], dtype='int'),
            'fgp_cols': np.arange(X.shape[1], dtype='int')
        }
