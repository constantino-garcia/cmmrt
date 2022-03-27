import argparse
import pickle

import numpy as np
import pandas as pd


def create_parser():
    parser = argparse.ArgumentParser(description="predict with DNN")
    parser.add_argument('--fingerprints', type=str, help='CSV with the fingerprints to predict', required=True)
    parser.add_argument('--dnn', type=str, help='pickled dnn', required=True)
    parser.add_argument('--preproc', type=str, help='pickled preproc', required=True)
    parser.add_argument('--save_to', type=str, help='save resulting dataframe to this CSV file', required=True)
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    fingerprints = pd.read_csv(args.fingerprints, dtype={'pid': str})
    pid = fingerprints.pid.values
    cmm_id = fingerprints.CMM_id.values
    fingerprints = fingerprints[[col for col in fingerprints.columns if col not in ['pid', 'CMM_id']]].values
    fingerprints = fingerprints.astype(np.float32)

    with open(args.preproc, 'rb') as f:
        preprocessor = pickle.load(f)
    # Need to modify the fgp_cols in the preprocessor, which were used to separate descriptors vs fingerpints. Now
    # we are just using fingerprints.
    preprocessor.fgp_cols = np.arange(fingerprints.shape[1])

    with open(args.dnn, 'rb') as f:
        dnn = pickle.load(f)

    X_preprocessed = preprocessor.transform(fingerprints)
    predictions = dnn.predict(X_preprocessed)

    results = pd.DataFrame({
        'pid': pid,
        'cmm_id': cmm_id,
        'prediction': predictions
    })
    results.to_csv(args.save_to, index=False)
