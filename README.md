# CMM-RT
This code implements methods for the accurate prediction of Retention Times 
(RTs) for a given Chromatographic Method (CM) using machine learning, as 
described in:

> García, C.A., Gil-de-la-Fuente A., Domingo-Almenara X., Barbas C., Otero, A. **Probabilistic metabolite annotation using retention time prediction and meta-learned projections** (Under review).



Highlights: 
* We have trained state-of-the-art machine learning regressors using the 80,038 
experimental RTs from the METLIN small molecule dataset (SMRT); both retained 
and unretained molecules were considered.
* 5,666 molecular descriptors and 2,214 fingerprints (MACCS166, Extended Connectivity, and Path Fingerprints fingerprints) were generated with
 the alvaDesc software. The models were trained using only the descriptors, 
 only the fingerprints, and both types of features simultaneously. Results suggest
 that fingerprints tend to perform better.
* The best results were obtained by a heavily regularized DNN trained with 
cosine annealing warm restarts and stochastic weight averaging, achieving a 
mean and median absolute errors of 39.2±1.2 s and 17.2 ± 0.9 s, respectively.
* A novel Bayesian meta-learning approach is proposed for RT projection between
 CMs from as few as 10 molecules while still obtaining competitive error rates 
 compared with previous approaches.
* We illustrate how the proposed DNN+meta-learned projections can be integrated into a 
metabolite annotation workflow. Indeed, we plan to integrate such approach into [CEU Mass Mediator](http://ceumass.eps.uspceu.es/).
 
Note that, to integrate the proposal into the CEU Mass Mediator platform, the code in this 
repository will continue to be developed. Hence, branch `paper` should be used as reference 
for reproducing the results of the paper. 

## Notebooks 
Notebooks illustrating several aspects of the tool are available under the `notebooks` folder:
* `train_with_rdkit.ipynb`: train a DNN with the methods of the paper on SMRT using RDKit fingerprints.
* `projections_to_different_cm.ipynb`: map experimental retention times to the retention times predicted with a DNN (or viceversa). This is done by training a meta-learned GP prior on a small subset of known molecules. 

## Fingerprints generation
To train your own model or to predict the RT of your own set of compounds it is necessary to generate the fingerprints using alvaDesc software (under license, check [alvadesc software](https://www.alvascience.com/alvadesc/)). 

You may find useful the files:

* [build_data.py](cmmrt/rt/build_data_cmm.py): it contains the necessary functions to generate fingerprints and/or descriptors. Specifically, the function generate_vector_fingerprints(aDesc, chemicalStructureFile = None, smiles = None) generates the fingerprints used in the CMM RT model. This function processes an instance of the alvaDesc software, and the input file path representing the compound of interest in the format SMILES, mol, SDF, mol2 or hin. If a String of SMILES is desired directly, it can be directly specified in the parameter SMILES. It returns a string formed the values of the ECFP, MACCSFP and PFP fingerprints sequentially joint. 
The function generate_vector_fps_descs(aDesc, chemicalStructureFile, fingerprint_types = ("ECFP", "MACCSFP", "PFP"), descriptors = True) generates both the the descriptors and the fingerprints. It contains the descriptors values value by value and the ECFP, MACCSFP and PFP fingerprints andsequentially joint. 
* [build_data_smrt.py](cmmrt/rt/build_data_smrt.py): it is an example of the processing of a CSV input file containing the pubchem id and the inchi. The pubchem id is used as a reference to access the corresponding SDF file representing the structure. It generates a file containing the ECFP, MACCSFP and PFP fingerprints sequentially joint for each input compound.
* [build_data_cmm.py](cmmrt/rt/build_data_cmm.py): it is an example of the processing of a CSV input file containing the pubchem id and the SMILES. The pubchem id is used as a reference to access the corresponding SDF file representing the structure. It generates a file containing the ECFP, MACCSFP and PFP fingerprints sequentially joint for each input compound.


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


