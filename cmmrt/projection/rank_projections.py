import copy
import os
import pickle

import gpytorch
import matplotlib.pyplot as plt
import pandas as pd
import torch

from cmmrt.projection.metalearning_train import get_representatives
from cmmrt.projection.data import Detrender


PATH = "results/projection/with_mean/detrender/wd_4em3_128"
N_GROUPS = 5

def get_ppm_error(mass, ppm_error=10):
    return (round(mass) * ppm_error) / 10 ** 6

def rank_projections(system, cuttoff, kegg_predret, kegg_to_rank, path,
                     n_annotations, n_groups=10, do_plot=False):
    system_data = kegg_predret[(kegg_predret.System == system) & (kegg_predret.rt > cuttoff)]
    model, scaler = pickle.load(open(os.path.join(path, f"gp_model_excluding_{system}.pkl"), "rb"))
    # print(model.gp.covar_module.kernels[0].base_kernel.lengthscale)
    system_data["rt"] = scaler.transform(system_data["rt"].values.reshape(-1, 1)).flatten()
    system_data["model_prediction"] = scaler.transform(system_data["model_prediction"].values.reshape(-1, 1)).flatten()
    system_data = system_data.astype({'model_prediction': 'float32', 'rt': 'float32'}, copy=False)
    test = copy.deepcopy(kegg_to_rank)
    test["model_prediction"] = scaler.transform(test["model_prediction"].values.reshape(-1, 1)).flatten()
    test = test.astype({'model_prediction': 'float32'}, copy=False)
    _, pubchems = get_representatives(system_data.model_prediction.values, system_data.Pubchem.values, n_annotations,
                                      n_groups)
    train_logical = system_data['Pubchem'].isin(pubchems)
    train = system_data[train_logical]
    # test = system_data[~train_logical]
    x, y = (
        torch.from_numpy(train.model_prediction.values.reshape(-1, 1)),
        torch.from_numpy(train.rt.values)
    )
    use_detrender = not ('no_detrender' in path)
    if use_detrender:
        detrender = Detrender()
        y = detrender.fit_transform(x, y)
    model.set_train_data(x, y)
    # model.train()
    # likelihood_param = [p for n, p in model.named_parameters() if 'likelihood' in n]
    # optimizer = torch.optim.Adam(likelihood_param, lr=0.1)
    # mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.gp.likelihood, model.gp)
    # training_iter = 500
    # for i in range(training_iter):
    #     optimizer.zero_grad()
    #     loss = -mll(model(x), y)
    #     loss.backward()
    #     if i % 100 == 0:
    #         print('Iter %d/%d - Loss: %.3f - noise: %.3f' % (
    #             i + 1, training_iter, loss.item(),
    #             model.gp.likelihood.noise.item()
    #         ))
    #     optimizer.step()
    model.eval()
    with torch.no_grad():
        predictions = model(torch.from_numpy(test.model_prediction.values.reshape(-1, 1)))
        # test["projection_mean"] = detrender.inverse_transform(predictions.mean)
        # test["projection_std"] = torch.sqrt(detrender.inverse_var_transform(predictions.variance))
        test["projection_mean"] = predictions.mean
        test["projection_std"] = torch.sqrt(predictions.variance)
        if use_detrender:
            test["projection_mean"] = detrender.inverse_transform(test["projection_mean"].values)
            test["projection_std"] = torch.sqrt(
                torch.from_numpy(detrender.inverse_var_transform(test["projection_std"].values**2))
            )
    in_top_results = []
    for index, row in system_data.iterrows():
        # Skip if the compound is not in the test set (since it wouldn't have a chance to be in the top results)
        if not row.Pubchem in test.Pubchem.values:
            continue
        error = get_ppm_error(row.mmass)
        candidates = test[(test["mmass"] >= (row.mmass - error)) & (test["mmass"] <= (row.mmass + error))]
        # print(row.Pubchem, "has ", candidates.shape[0], " candidates")
        if candidates.shape[0] > 3:
            candidates["z_score"] = abs(row.rt - candidates.projection_mean) / candidates.projection_std
            candidates.sort_values("z_score", inplace=True)
            in_top_results.append([row["Pubchem"] in candidates.iloc[:k].Pubchem.values for k in range(1, 4)])

    if do_plot:
        sorted_x = torch.arange(x.min() - 0.5, x.max() + 0.5, 0.01, dtype=torch.float32)
        with torch.no_grad():
            mean = model.gp.mean_module(sorted_x)
            var = model.gp.covar_module(sorted_x,diag=True)
            preds = model(sorted_x)
            preds_mean = preds.mean
        if use_detrender:
            y = detrender.inverse_transform(y)
            mean = detrender.inverse_transform(mean)
            var = detrender.inverse_var_transform(var)
            preds_mean = detrender.inverse_transform(preds_mean)
        plt.scatter(system_data.model_prediction.values,
                    system_data.rt.values)
        plt.scatter(x, y, marker='x')
        plt.plot(sorted_x, preds_mean)
        plt.plot(sorted_x, mean, '--', color='orange')
        plt.plot(sorted_x, mean + 2 * torch.sqrt(var), '--', color='orange')
        plt.plot(sorted_x, mean - 2 * torch.sqrt(var), '--', color='orange')
        plt.title(system)
        plt.show()

    in_top_df = pd.DataFrame(in_top_results).rename(columns={0: "in_top_1", 1: "in_top_2", 2: "in_top_3"})
    return in_top_df


if __name__ == "__main__":
    path = PATH
    n_annotations_list = [10, 20, 25, 30, 40, 50]
    n_repeats = 500
    n_groups = N_GROUPS

    kegg_predret = pd.read_csv("rt_data/kegg_predret.csv").drop("Unnamed: 0", axis=1)
    kegg_to_rank = pd.read_csv("rt_data/kegg_molecules_to_rank.csv").drop("Unnamed: 0", axis=1)

    in_top_df_list = []
    systems = [("FEM_long", 5), ("LIFE_old", 1), ("FEM_orbitrap_plasma", 2), ("RIKEN", 1)]
    for system, cuttoff in systems:
        for n_annotations in n_annotations_list:
            for n_rep in range(n_repeats):
                do_plot = (n_annotations == 50) and (n_rep == 0)
                do_plot = (n_rep == 0)
                in_top_df = rank_projections(system, cuttoff, kegg_predret, kegg_to_rank,
                                             path, n_annotations, n_groups, do_plot=do_plot)
                in_top_df = pd.DataFrame(in_top_df.mean(axis=0)).transpose()
                in_top_df['n_annotations'] = n_annotations
                in_top_df['excluded_system'] = system
                in_top_df['repetition_nb'] = n_rep
                in_top_df_list.append(in_top_df)

    rank_projections_df = pd.concat(in_top_df_list)
    rank_projections_df.to_csv(os.path.join(path, "rankings.csv"))
    print(rank_projections_df.groupby(["excluded_system", "n_annotations"]).mean())
    print(rank_projections_df.drop(["repetition_nb"], axis=1).
          groupby(["excluded_system", "n_annotations"]).agg(['mean', 'std']))

    print('done')
