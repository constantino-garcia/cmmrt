import os
import tempfile

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.model_selection import StratifiedKFold

from cmmrt.utils.train.model_selection import stratify_y

from validate_model import create_cv_parser
from train_dnn_model import load_data_and_configs, tune_and_fit


if __name__ == '__main__':
    N_STRATS = 6
    parser = create_cv_parser(
        description="Cross-validate Blender", default_storage="sqlite:///cv.db", default_study="cv"
    )
    args = parser.parse_args()

    data, param_search_config = load_data_and_configs(args, download_directory="rt_data")

    cv_folds = 2 if args.smoke_test else args.cv_folds
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=args.random_state + 500)
    strats = stratify_y(data.y, N_STRATS)

    metrics = {'mae': mean_absolute_error, 'medae': median_absolute_error}

    base_study_name = param_search_config.study_prefix
    results = []
    for fold, (train_index, test_index) in enumerate(cv.split(data.X, strats)):
        # FIXME: should replace namedTuples with recordtype to avoid next line
        param_search_config = param_search_config._replace(study_prefix=base_study_name + f"-fold-{fold}")
        train_data = data[train_index]
        test_data = data[test_index]

        preprocessor, dnn = (
            tune_and_fit(train_data, param_search_config=param_search_config)
        )
        predictions = dnn.predict(preprocessor.transform(test_data.X))
        performance_dict = {k: metric(test_data.y, predictions) for k, metric in metrics.items()}
        performance_dict.update({'fold': fold})
        results.append(performance_dict)

    results = pd.DataFrame(results)

    print(f"Saving results to {args.csv_output}")
    print(results)
    results.to_csv(args.csv_output, index=False)
