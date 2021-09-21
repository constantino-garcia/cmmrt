# CMM-RT
This code implements methods for the accurate prediction of Retention Times 
(RTs) for a given Chromatographic Method (CM) using machine learning, as 
described in:

> García, C.A., Gil-de-la-Fuente A., Domingo-Almenara X., Barbas C., Otero, A. **Evaluation of machine learning techniques for
small molecule retention time prediction** *(Under review)*.

Highlights: 
* We have trained state-of-the-art machine learning regressors using the 80,038 
experimental RTs from the METLIN small molecule dataset (SMRT); both retained 
and unretained molecules were considered.
* 5666 molecular descriptors and 2214 fingerprints (MACCS166, 
Extended Connectivity, and Path Fingerprints fingerprints) were generated with
 the alvaDesc software. The models were trained using only the descriptors, 
 only the fingerprints, and both types of features simultaneously. Results suggest
 that fingerprints tend to perform better.
* The best results were obtained by a heavily regularized DNN trained with 
cosine annealing warm restarts and stochastic weight averaging, achieving a 
mean and median absolute errors of 39.2±1.2 s and 17.2 ± 0.9 s, respectively.
* A novel Bayesian meta-learning approach is proposed for RT projection between
 CMs from as few as 20 molecules while still obtaining competitive error rates 
 compared with previous approaches.


## Getting started
Clone the project and use the `Makefile` to get started with the project. You 
may use `make help` to list available rules. The most important ones are 
```bash
# Available rules:

# install             Install Python Dependencies and cmmrt package
# test_predictor      Test the performance of all RT predictors using nested cross-validation
# test_projections    Test the performance of meta-training for projections using 4 reference CMs
# train_predictor     Train all RT predictors using hyperparameter tuning
# train_projections   Meta-train a GP for projections using all data from PredRet database
```

Hence, to install the `cmmrt` package (and its dependencies) use 
```bash
make install
```

You may run some **smoke tests** using the `Makefile` to get started with the 
code. Note that, as part of the code, data is automatically downloaded for you. 
You may run:
* Prediction experiments:
    * `make train_predictor`: Trains a subset of regressors on a subset of the 
    SMRT database using Bayesian hyperparameter search. The best resulting blender
    (which contains all other regressors) is saved to the `saved_models` folder.
    A summary of all tested models is stored in the database `results/optuna/train.db`.
    * `make test_predictor`: Tests the performance of a subset of regressors on
    a subset of the SMRT database using nested cross-validation. A summary of 
    the results is stored in the CSV `results/rt/rt_cv.csv`. A summary of all tested 
    models during hyperparameter search is stored in `results/optuna/cv.db` (nested
    within the outer loop).
* Projection experiments: 
    * `make train_projections`: Meta-trains a Gaussian Process (GP) for computing 
    projections between CMs using the PredRet database for a few epochs. The
    resulting GP is stored in the `saved_models` folder.
    * `make test_projections`: Tests the performance of GP meta-training for 
    computing projections between CMs. Four CMs of reference are used for the
    test: FEM long, LIFE old, FEM orbitrap plasma and RIKEN. For each system, 
    the GP is meta-trained using all other CMs from the PredRet database before
    testing the performance on the target CM. As in the other smoke tests, 
    meta-training only proceeds for a few epochs. CSVs and figures summarizing
    the projections are stored under `results/projection`.

## Customizing experiments
As previously stated, the `Makefile` only runs quick tests to avoid waiting 
for hours before getting any result. However, you may easily modify the `Makefile`
(or use it as reference) for running your own experiments. For example, the rule
`train_predictor` is:
```makefile
# Makefile
# ...
train_predictor:
	$(PYTHON_INTERPRETER) cmmrt/rt/train_model.py \
		--storage sqlite:///results/optuna/train.db --save_to saved_models \
		--smoke_test # FIXME: remove this line for complete training
# ...
```
As noted in the `Makefile`, you may remove the last line for complete training, 
using all regressors and the complete SMRT dataset. Furthermore, note that 
you may change the database and/or folder storing the results. Indeed, unless
you really understand how to reuse partial runs, we recommend not reusing 
hyperparameter's databases.

Finally, note that you may find additional options for running your experiments
by consulting the `help` option of the Python scripts. E.g.:
```bash
$(PYTHON_INTERPRETER) cmmrt/rt/train_model.py --help
# usage: train_model.py [-h] [--storage STORAGE] [--study STUDY] [--train_size TRAIN_SIZE]
#                       [--param_search_folds PARAM_SEARCH_FOLDS] [--trials TRIALS] [--smoke_test]
#                       [--random_state RANDOM_STATE] [--save_to SAVE_TO]
# 
# Train blender and all base-models
# 
# optional arguments:
#   -h, --help            show this help message and exit
#   --storage STORAGE     SQLITE DB for storing the results of param search (e.g.: sqlite:///train.db
#   --study STUDY         Study name to identify param search results withing the DB
#   --train_size TRAIN_SIZE
#                         Percentage of the training set to train the base classifiers. The remainder
#                         is used to train the meta-classifier
#   --param_search_folds PARAM_SEARCH_FOLDS
#                         Number of folds to be used in param search
#   --trials TRIALS       Number of trials in param search
#   --smoke_test          Use small model and subsample training data for quick testing.
#                         param_search_folds and trials are also overridden
#   --random_state RANDOM_STATE
#                         Random state for reproducibility or reusing param search results
#   --save_to SAVE_TO     folder where to save the preprocessor and regressor models
```


