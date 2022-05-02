"""
This script is work in progress... To make it work, you need to:
1. Create models using metalearning_test.py. Run
        $ make test_projections
    in an OS terminal.
2. Run this script. Run
        $ make rank_candidates
    in an OS terminal.
"""
import copy
import os

import importlib_resources
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from cmmrt.projection.data import get_representatives
from cmmrt.rt.predictions import load_cmm_predictions
from cmmrt.projection.models.projector.loader import _load_projector_pipeline_from

PATH = "results/projection/models_to_rank_with"
N_GROUPS = 4

np.random.seed(15062021)


def get_ppm_error(mass, ppm_error=10):
    return (round(mass) * ppm_error) / 10 ** 6


def rank_projections(system, cuttoff, kegg_predret, kegg_to_rank,
                     n_annotations, n_groups=10, do_plot=False, mass_error_seed=0):
    system_data = kegg_predret[(kegg_predret.System == system) & (kegg_predret.rt > cuttoff)]
    system_data = system_data.astype({'model_prediction': 'float32', 'rt': 'float32'}, copy=False)
    test = copy.deepcopy(kegg_to_rank)
    test = test.astype({'model_prediction': 'float32'}, copy=False)
    # TODO: avoid the use of _ function
    projector = _load_projector_pipeline_from(f"cmmrt/data/metalearned_projectors/{system}.pt", mean='constant',
                                              kernel='poly')
    _, pubchems = get_representatives(system_data.model_prediction.values, system_data.Pubchem.values,
                                      n_annotations, n_groups)
    train_logical = system_data['Pubchem'].isin(pubchems)
    train = system_data[train_logical]
    x, y = (
        torch.from_numpy(train.model_prediction.values.reshape(-1, 1)),
        torch.from_numpy(train.rt.values)
    )
    projector.projector.prepare_metatesting()
    projector.fit(x, y)
    mass_filtering_results = []
    in_top_results = []
    # Set seed for random mass errors
    if mass_error_seed is not None:
        np.random.seed(mass_error_seed)
    for index, row in system_data.iterrows():
        # Skip if the compound is not in the test set (since it wouldn't have a chance to be in the top results)
        if not row.Pubchem in test.Pubchem.values:
            continue
        error = get_ppm_error(row.mmass)
        # (error / 3) to ensure that 99.7% of experimental mass is within +/- 10 ppm
        experimental_mass = row.mmass + (error / 3) * np.random.randn()
        # Clip the values to ensure that the true molecule is between candidates after mass search.
        experimental_mass = np.clip(experimental_mass, row.mmass - error, row.mmass + error)
        search_window = get_ppm_error(experimental_mass)
        candidates = test[
            (test["mmass"] >= (experimental_mass - search_window))
            & (test["mmass"] <= (experimental_mass + search_window))
            ].copy()

        # print(row.Pubchem, "has ", candidates.shape[0], " candidates")
        if candidates.shape[0] > 3:
            candidates['z_score'] = pd.NA
            candidates['mass_error'] = abs(candidates.mmass - experimental_mass)
            # add small noise to unbreak ties
            candidates['mass_error'] = candidates['mass_error'] + np.random.uniform(0, 1e-6, candidates.shape[0])
            candidates.sort_values(by='mass_error', inplace=True)
            mass_filtering_results.append(
                [row["Pubchem"] in candidates.iloc[:k].Pubchem.values for k in range(1, 4)]
            )
            scores = projector.z_score(candidates[['model_prediction']].values, np.array([row.rt]))
            scores = scores.cpu().numpy()
            candidates.loc[:, 'z_score'] = scores
            candidates.sort_values("z_score", inplace=True)
            in_top_results.append([row["Pubchem"] in candidates.iloc[:k].Pubchem.values for k in range(1, 4)])

    if do_plot:
        sorted_x = torch.arange(x.min() - 0.5, x.max() + 0.5, 0.1, dtype=torch.float32)
        plt.scatter(system_data.model_prediction.values,
                    system_data.rt.values)
        plt.scatter(x, y, marker='x')
        preds_mean, lb, ub = projector.predict(sorted_x)
        plt.fill_between(sorted_x, lb, ub, alpha=0.2, color='orange')
        plt.plot(sorted_x, preds_mean, color='orange')
        with torch.no_grad():
            sorted_x_ = torch.from_numpy(projector.x_scaler.transform(sorted_x.numpy().reshape(-1, 1)))
            tmp = projector.projector.gp.mean_module(sorted_x_)
            tmp = projector.y_scaler.inverse_transform(tmp.reshape(-1, 1)).flatten()
            plt.plot(sorted_x, tmp, color='green')
        plt.title(system)
        plt.show()

    in_top_df = pd.DataFrame(in_top_results).rename(columns={0: "in_top_1", 1: "in_top_2", 2: "in_top_3"})
    in_top_df_mass_filtering = pd.DataFrame(mass_filtering_results).rename(
        columns={0: "in_top_1", 1: "in_top_2", 2: "in_top_3"})
    return in_top_df, in_top_df_mass_filtering


def load_kegg_experiment_data():
    preds = load_cmm_predictions().drop(columns=['cmm_id'])
    # same pubchem but different predictions
    tmp = preds.groupby('Pubchem').mean()
    preds = pd.DataFrame({
        'Pubchem': tmp.index.values,
        'model_prediction': tmp.rt_pred.values
    })
    preds.model_prediction /= 60

    kegg_predret_path = importlib_resources.files("cmmrt.data.rank_experiment").joinpath("kegg_predret.csv")
    kegg_predret = pd.read_csv(kegg_predret_path).drop(columns=['Unnamed: 0', 'model_prediction'])
    to_rank_path = importlib_resources.files("cmmrt.data.rank_experiment").joinpath("kegg_molecules_to_rank.csv")
    to_rank = pd.read_csv(to_rank_path).drop(columns=['Unnamed: 0', 'model_prediction'])

    kegg_predret_cmm = kegg_predret.merge(preds, on='Pubchem', how='inner').drop_duplicates(ignore_index=True)
    to_rank_cmm = to_rank.merge(preds, on='Pubchem', how='inner').drop_duplicates(ignore_index=True)

    return kegg_predret_cmm, to_rank_cmm


if __name__ == "__main__":
    np.random.seed(124245)
    n_annotations_list = [10, 20, 30, 40, 50]

    n_repeats = 10
    n_groups = N_GROUPS

    kegg_predret, kegg_to_rank = load_kegg_experiment_data()

    in_top_df_list = []
    in_top_df_mass_filtering_list = []
    systems = [("FEM_long", 5), ("LIFE_old", 1), ("FEM_orbitrap_plasma", 2), ("RIKEN", 1)]
    mass_error_seeds = np.random.randint(0, 10000000, size=n_repeats)
    for system, cuttoff in systems:
        for n_annotations in n_annotations_list:
            for n_rep in tqdm(range(n_repeats)):
                # do_plot = (n_annotations == 50) and (n_rep == 0)
                do_plot = False
                in_top_df, in_top_df_mass_filtering = rank_projections(system, cuttoff, kegg_predret, kegg_to_rank,
                                                                       n_annotations, n_groups, do_plot=do_plot,
                                                                       mass_error_seed=mass_error_seeds[n_rep])


                def summarise_in_top(df):
                    in_top_df = pd.DataFrame(df.mean(axis=0)).transpose()
                    in_top_df['n_annotations'] = n_annotations
                    in_top_df['excluded_system'] = system
                    in_top_df['repetition_nb'] = n_rep
                    return in_top_df


                in_top_df_list.append(summarise_in_top(in_top_df))
                in_top_df_mass_filtering_list.append(summarise_in_top(in_top_df_mass_filtering))

    rank_projections_df = pd.concat(in_top_df_list)
    rank_projections_df.to_csv(os.path.join(PATH, "rankings.csv"))

    rank_projections_mass_filtering_df = pd.concat(in_top_df_mass_filtering_list)
    rank_projections_mass_filtering_df.to_csv(os.path.join(PATH, "rankings_mass_filtering_only.csv"))

    print(rank_projections_df.groupby(["excluded_system", "n_annotations"]).mean())
    print(rank_projections_df.drop(["repetition_nb"], axis=1).
          groupby(["excluded_system", "n_annotations"]).agg(['mean', 'std']))

    print('done')
