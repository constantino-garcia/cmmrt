"""
Models used to predict retention times using Alvadesc's fingerprints and descriptors.

This module implements models for predicting retention times using Alvadesc's fingerprints and descriptors. Final models
implement the scikit-learn interface. Models include:
* preprocessor.Preprocessor: scikit-learn transformer able to select fingerprints, descriptors or both; eliminate variables with
low variance; eliminate correlated variables; and add an indicator column for non-retained molecules (based on
XGBoost classifier).
* gp.DKL: Gaussian process with deep kernel (deep kernel learning)
* nn.SkDnn: Deep Neural network trained with warm restarts and Stochastic Weight averaging (SWA).
* Gradient Boosting Machines:
    * gbm.xgboost.WeightedXGBClassifier: XGBoost classifier assigning different weights to retained and non-retained molecules.
    * gbm.xgboost.SelectiveXGBRegressor: XGBoost regressor with extra capabilities to select and eliminate features.
    * gbm.WeightedCatBoostRegressor: CatBoost regressor assigning different weights to retained and non-retained molecules.
* ensemble.Blender: An ensemble of the previous models based on blending.
"""