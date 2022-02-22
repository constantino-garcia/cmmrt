import bz2
import copy
import os
import pickle

import numpy as np
import pandas as pd


class AlvadescDataset:
    def __init__(self, download_directory="rt_data"):
        filename = os.path.join(download_directory, "alvadesc.pklz")
        if not os.path.exists(filename):
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

            self.X = np.concatenate([X_desc, X_fgp], axis=1)
            self.desc_cols = np.arange(X_desc.shape[1], dtype='int')
            self.fgp_cols = np.arange(X_desc.shape[1], self.X.shape[1], dtype='int')
            self.y = descriptors_fgp['rt'].values.flatten()

            print('saving')
            with bz2.BZ2File(filename, "wb") as f:
                pickle.dump([self.X, self.y, self.desc_cols, self.fgp_cols], f)
        else:
            with bz2.BZ2File(filename, "rb") as f:
                self.X, self.y, self.desc_cols, self.fgp_cols = pickle.load(f)
        self.X = self.X.astype('float32')
        self.y = self.y.astype('float32')

    @property
    def fingerprints(self):
        return self.X[:, self.fgp_cols]

    @property
    def descriptors(self):
        return self.X[:, self.desc_cols]

    def __getitem__(self, item):
        self_copy = copy.deepcopy(self)
        self_copy.X = self_copy.X[item, :]
        self_copy.y = self_copy.y[item]
        return self_copy


def load_alvadesc_descriptors(download_directory="rt_data", n=None, split_as_np=True):
    filename = "descriptors.pklz"
    filename = os.path.join(download_directory, filename)
    url = "https://drive.google.com/u/0/uc?id=1MMPsk8jghXfzp2DvrRsHsz4iB03shndP&export=download"
    return _load_pickled_data(filename, url, n, split_as_np)


def load_alvadesc_fingerprints(download_directory="rt_data", n=None, split_as_np=True):
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


def is_non_retained(y):
    return (y < 300).astype('int')


def is_binary_feature(x):
    ux = np.unique(x)
    if len(ux) == 1:
        return ux == 0 or ux == 1
    if len(ux) == 2:
        return np.all(np.sort(ux) == np.array([0, 1]))
    else:
        return False


def binary_features_cols(X):
    return np.where(np.apply_along_axis(is_binary_feature, 0, X))[0]
