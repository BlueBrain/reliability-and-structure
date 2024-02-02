"""
Plotting oracle score (kind of reliability) against simplex dim (and position within simplex) for MICrONS
author: András Ecker, last update: 02.2024
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import sys
sys.path.append("../../../library")
from coupling_coefficient import zscore_loo_ccs
from structural_basic import *
import conntility
from connalysis.network import stats as nstats
from connalysis.network import topology
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="ticks", context="paper", font="Helvetica Neue",
        rc={"axes.labelsize": 7, "legend.fontsize": 6, "axes.linewidth": 0.6, "xtick.labelsize": 6, "ytick.labelsize": 6,
            "xtick.major.size": 2, "xtick.major.width": 0.5, "xtick.minor.size": 1.5, "xtick.minor.width": 0.3,
            "ytick.major.size": 2, "ytick.major.width": 0.5, "ytick.minor.size": 1.5, "ytick.minor.width": 0.3})
STRUCTURAL_DATA_DIR = "/gpfs/bbp.cscs.ch/home/egassant/reliability_and_structure/data_analysis/data"
FUNCTIONAL_DATA_DIR = "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/assemblyfire/MICrONS"
LAYER_DICT = {"23P": 23, "4P": 4, "5P_IT": 5, "5P_NP": 5, "5P_PT": 5, "6IT": 6, "6CT": 6}
L234_MTYPES = ["23P", "4P"]
L56_MTYPES = ["5P_IT", "5P_NP", "5P_PT", "6IT", "6CT"]
COLORS = {"all": "tab:orange", "sink": "tab:green", "source": "tab:blue"}
MARKERS = {"all": "o", "sink": ">", "source":"<"}


def get_functional_data(npzf_name, conn_mat):
    """Loads ''spikes'' and oracle score from MICrONS functional data (saved to npz in `assemblyfire`),
    maps idx to the structural connectome, calculates coupling coefficients
    and padds everything with nans to have the same shape (and index) as the structural data"""
    tmp = np.load(npzf_name)
    idx, spikes = tmp["idx"], tmp["spikes"]
    # drop duplicated idx and get oracle scores
    unique_idx, counts = np.unique(idx, return_counts=True)
    idx_tmp = np.in1d(idx, unique_idx[counts == 1])
    idx, spikes = idx[idx_tmp], spikes[idx_tmp, :]
    oracle_scores = pd.Series(tmp["oracle_scores"][idx_tmp], index=idx)
    oracle_scores = oracle_scores.loc[oracle_scores.notna()]
    # match idx to structural data
    valid_idx = conn_mat.vertices.id[np.isin(conn_mat.vertices.id, oracle_scores.index)]
    oracle_scores = oracle_scores.loc[oracle_scores.index.isin(valid_idx)]
    valid_idx_tmp = pd.Series(valid_idx.index.to_numpy(), index=valid_idx.to_numpy())
    oracle_scores = pd.Series(oracle_scores.to_numpy(), index=valid_idx_tmp.loc[oracle_scores.index].to_numpy()).sort_index()
    # index out spikes as well (and sort corresponding gids)
    idx_tmp = np.where(np.in1d(idx, valid_idx_tmp.index.to_numpy()))[0]
    gids = valid_idx_tmp.loc[idx[idx_tmp]].to_numpy()
    sort_idx = np.argsort(gids)
    spikes, gids = spikes[idx_tmp[sort_idx], :], gids[sort_idx]
    # get rates and coupling coefficients
    rates = pd.Series(np.mean(spikes, axis=1), index=gids)
    ccs = pd.Series(zscore_loo_ccs(spikes), index=gids)
    # padd everything with NaNs
    idx = conn_mat.vertices.index.to_numpy()
    data = np.full(len(idx), np.nan)
    data[rates.index.to_numpy()] = rates.to_numpy()
    rates = pd.Series(data, index=idx, name="rate")
    data = np.full(len(idx), np.nan)
    data[oracle_scores.index.to_numpy()] = oracle_scores.to_numpy()
    oracle_scores = pd.Series(data, index=idx, name="oracle_score")
    data = np.full(len(idx), np.nan)
    data[ccs.index.to_numpy()] = ccs.to_numpy()
    ccs = pd.Series(data, index=idx, name="coupling_coeff")
    return rates, oracle_scores, ccs


def load_functional_data(session_idx, scan_idx, conn_mat, pklf_name=None):
    """Either loads save DataFrame or calls `get_functional_data()` from above across scans"""
    if pklf_name is None or (pklf_name is not None and not os.path.isfile(pklf_name)):
        dfs = []
        for session_id, scan_id in zip(session_idx, scan_idx):
            name_tag = "session%i_scan%i" % (session_id, scan_id)
            rates, oss, ccs = get_functional_data(os.path.join(FUNCTIONAL_DATA_DIR, "MICrONS_%s.npz" % name_tag), conn_mat)
            df = pd.concat([rates, oss, ccs], axis=1)
            df.columns = pd.MultiIndex.from_arrays([np.full(3, name_tag), df.columns.to_numpy()])
            dfs.append(df)
        df = pd.concat(dfs, axis=1)
        if pklf_name is not None:
            df.to_pickle(pklf_name)
    else:
        df = pd.read_pickle(pklf_name)
    return df


def load_simplex_list(maximal=True):
    """Loads saved simplex list (for MICrONS)"""
    str = "_maximal" if maximal else "_"
    return pd.read_pickle(os.path.join(STRUCTURAL_DATA_DIR, "MICrONS_list_simplices_by_dimension%s.pkl" % str))["original"]


def agg_along_dims(stats):
    """Aggregates results across dimensions (indexes out source, sinke and all positions)"""
    dims = stats.keys()
    df = {"all": {}, "sink": {}, "source": {}}
    for dim in dims:
        mean = stats[dim]["mean"]
        err = stats[dim]["sem"]
        df["all"][dim] = [mean.loc["all"], err.loc["all"]]
        df["source"][dim] = [mean.iloc[0], err.iloc[0]]
        df["sink"][dim] = [mean.iloc[-2], err.iloc[-2]]
    return {key: pd.DataFrame.from_dict(df[key], orient="index", columns=["mean", "sem"])
            for key in df.keys()}


def plot_lw_rates(df, fig_name):
    """Plots layer-wise distribution of firing rates"""
    fig = plt.figure(figsize=(1.8, 1.6))
    ax = fig.add_subplot(1, 1, 1)
    sns.boxplot(x="layer", y="rate", order=[23, 4, 5, 6], linewidth=0.5, fliersize=0, data=df, ax=ax)
    sns.stripplot(x="layer", y="rate", order=[23, 4, 5, 6], color="black", dodge=True, size=1., jitter=0.1, data=df, ax=ax)
    sns.despine(trim=True, offset=1)
    fig.savefig(fig_name, bbox_inches="tight", transparent=True)
    plt.close(fig)


def plot_oracle_score_vs_x(df, x, fig_name):
    """Scatter plot (and regression line) of anything (e.g. rate or CC) vs. oracle score"""
    fig = plt.figure(figsize=(1.8, 1.6))
    ax = fig.add_subplot(1, 1, 1)
    sns.regplot(x=x, y="oracle_score", marker='.', scatter_kws={"s": 7, "edgecolor": "none"},
                line_kws={"linewidth": 0.5}, data=df, ax=ax)
    ax.text(df[x].quantile(0.99), 0, "r = %.2f" % pearsonr(df[x].to_numpy(), df["oracle_score"].to_numpy()).statistic,
            horizontalalignment="center", fontsize=5)
    sns.despine(trim=True, offset=1)
    fig.savefig(fig_name, bbox_inches="tight", transparent=True)
    plt.close(fig)


def plot_y_vs_sdim(stats, node_part_sums, ylabel, fig_name, text_offset=0.01):
    """Plot anything (e.g. a functional feature like oracle score) vs. simplex dimension (and position)"""
    fig = plt.figure(figsize=(1.8, 1.6))
    ax = fig.add_subplot(1, 1, 1)
    for pos, color in COLORS.items():
        x = stats[pos].index.to_numpy()
        ax.plot(x, stats[pos]["mean"], marker=MARKERS[pos], label=pos)
        ax.fill_between(x, stats[pos]["mean"] - stats[pos]["sem"], stats[pos]["mean"] + stats[pos]["sem"], alpha=0.2)
    y = stats["all"]["mean"].max() + text_offset
    for x_ in x:
        ax.text(x_, y, node_part_sums[x_], horizontalalignment="center", rotation="vertical", fontsize=6)
    ax.set_xticks(x)
    ax.set_xlabel("Simplex dimension")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    sns.despine(trim=True, offset=1)
    fig.savefig(fig_name, bbox_inches="tight", transparent=True)
    plt.close(fig)


def plot_y_vs_sdim_summary(dict, ylabel, fig_name):
    """Plot anything (e.g. oracle score or node part sums) vs. simplex dimension across scans and sessions"""
    fig = plt.figure(figsize=(1.8, 1.6))
    ax = fig.add_subplot(1, 1, 1)
    x = np.arange(1, 8)  # pretty hard coded...
    ys = np.full((len(dict), len(x)), np.nan)
    for i, (label, y) in enumerate(dict.items()):
        ax.plot(x, y, label=label)
        ys[i, :] = y
    ax.plot(x, np.nanmean(ys, axis=0), color="black", label="mean of all scans")
    ax.set_xticks(x)
    ax.set_xlabel("Simplex dimension")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, bbox_to_anchor=(1., 1.1))
    sns.despine(trim=True, offset=1)
    fig.savefig(fig_name, bbox_inches="tight", transparent=True)
    plt.close(fig)


def main_functional(conn_mat, fn_df,  session_idx, scan_idx):
    dfs = []
    for session_id, scan_id in zip(session_idx, scan_idx):
        name_tag = "session%i_scan%i" % (session_id, scan_id)
        df = fn_df[name_tag]
        df = df.loc[df["rate"].notna()]
        df["mtype"] = conn_mat.vertices.loc[df.index, "cell_type"]
        df["layer"] = df["mtype"].map(LAYER_DICT)
        dfs.append(df)
        plot_lw_rates(df, "figs/MICrONS_%s_l-w_rate.pdf" % name_tag)
        plot_oracle_score_vs_x(df, "rate", "figs/MICrONS_%s_oracle_score_vs_rate.pdf" % name_tag)
        plot_oracle_score_vs_x(df, "coupling_coeff", "figs/MICrONS_%s_oracle_score_vs_coupling_coeff.pdf" % name_tag)
    df = pd.concat(dfs)
    mean_df = df.groupby(df.index)[["rate", "oracle_score", "coupling_coeff"]].agg("mean")
    mean_df["mtype"] = conn_mat.vertices.loc[mean_df.index, "cell_type"]
    mean_df["layer"] = mean_df["mtype"].map(LAYER_DICT)
    plot_lw_rates(df, "figs/MICrONS_l-w_rate.pdf")
    plot_oracle_score_vs_x(df, "rate", "figs/MICrONS_oracle_score_vs_rate.pdf")
    plot_oracle_score_vs_x(df, "coupling_coeff", "figs/MICrONS_oracle_score_vs_coupling_coeff.pdf")


def main(conn_mat, fn_df, session_idx, scan_idx, fn_feature="oracle_score", maximal=False, only_l234=False):
    max_str = "_max_" if maximal else "_"
    l234_str = "_L234_" if only_l234 else "_"
    simplices = load_simplex_list(maximal=maximal)
    node_part = topology.node_participation(conn_mat.matrix, max_simplices=maximal, threads=8)
    if only_l234:
        node_part.loc[conn_mat.vertices["cell_type"].isin(L56_MTYPES), :] = 0
    all_node_part_sums = node_part.sum()

    sum_stats, frac_node_parts = {}, {}
    for session_id, scan_id in zip(session_idx, scan_idx):
        name_tag = "session%i_scan%i" % (session_id, scan_id)
        functional_data = fn_df[name_tag, fn_feature]
        if only_l234:
            mtypes = conn_mat.vertices.loc[functional_data.loc[functional_data.notna()].index, "cell_type"]
            functional_data.loc[mtypes.loc[mtypes.isin(L56_MTYPES)].index] = np.nan
        stats = agg_along_dims(nstats.node_stats_per_position(simplices, functional_data,
                                                              dims=simplices.index.drop(0), with_multiplicity=True))
        sum_stats[name_tag] = stats["all"]["mean"].to_numpy()
        node_part_sums = node_part.loc[functional_data.notna()].sum()
        frac_node_parts[name_tag] = node_part_sums.loc[1:].to_numpy() / all_node_part_sums.loc[1:].to_numpy()
        plot_y_vs_sdim(stats, node_part_sums, "Mean %s" % fn_feature,
                       "figs/MICrONS_%s_%s%svs%ssimplex_dim.pdf" % (name_tag, fn_feature, l234_str, max_str))
    # summary plots across scans
    plot_y_vs_sdim_summary(sum_stats, "Mean %s" % fn_feature,
                           "figs/MICrONS_%s%svs%ssimplex_dim.pdf" % (fn_feature, l234_str, max_str))
    plot_y_vs_sdim_summary(frac_node_parts, "Node participation ratio",
                           "figs/MICrONS_node_part_sum%svs%ssimplex_dim.pdf" % (l234_str, max_str))


if __name__ == "__main__":
    session_idx = [4, 5, 6, 6, 6, 7]  # , 8, 9],
    scan_idx = [7, 7, 2, 4, 7, 4]  # , 5, 3]
    conn_mat = load_connectome(STRUCTURAL_DATA_DIR, "MICrONS")
    fn_df = load_functional_data(session_idx, scan_idx, conn_mat,
                                 pklf_name=os.path.join(FUNCTIONAL_DATA_DIR, "MICrONS_functional_summary.pkl"))
    main_functional(conn_mat, fn_df, session_idx, scan_idx)
    main(conn_mat, fn_df, session_idx, scan_idx, maximal=True, only_l234=True)
    main(conn_mat, fn_df, session_idx, scan_idx, maximal=True, only_l234=False)
    main(conn_mat, fn_df, session_idx, scan_idx, maximal=False, only_l234=True)
    main(conn_mat, fn_df, session_idx, scan_idx, maximal=False, only_l234=False)
    main(conn_mat, fn_df, session_idx, scan_idx, fn_feature="coupling_coeff", maximal=True, only_l234=True)
    main(conn_mat, fn_df, session_idx, scan_idx, fn_feature="coupling_coeff", maximal=True, only_l234=False)
    main(conn_mat, fn_df, session_idx, scan_idx, fn_feature="coupling_coeff", maximal=False, only_l234=True)
    main(conn_mat, fn_df, session_idx, scan_idx, fn_feature="coupling_coeff", maximal=False, only_l234=False)







