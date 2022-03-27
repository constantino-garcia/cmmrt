import importlib_resources
import pandas as pd


def load_cmm_predictions():
    # The following predictions were created using predict_with_dnn (see DNN branch) on CMM (CMM_predictions)
    # and a custom list of missing molecules available in Predret but not in the paper
    predictions = []
    # CMM_predictions: predictions of molecules in CMM database
    # missing_predictions: molecules from predret not available in CMM database
    # missing_predictions2: molecules from HMDB not available in CMM database
    for filename in ["CMM_predictions.csv", "missing_predictions.csv", "missing_predictions2.csv"]:
        path = importlib_resources.files("cmmrt.data.predictions").joinpath(filename)
        predictions.append(pd.read_csv(path))
    predictions = pd.concat(predictions)
    predictions.rename({'pid': 'Pubchem', 'prediction': 'rt_pred'}, axis=1, inplace=True)
    predictions = predictions[predictions['Pubchem'] != "\\N"]
    predictions = predictions.astype({'Pubchem': int})
    return predictions.drop_duplicates()
