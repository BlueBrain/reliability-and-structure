"""
Plotting oracle score (kind of reliability) against simplex dim (and position within simplex) for MICrONS
author: András Ecker, last update: 01.2024
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import sys
sys.path.append("../../../library")
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


def load_simplex_list(maximal=True):
    """Loads saved simplex list (for MICrONS)"""
    str = "_maximal" if maximal else "_"
    return pd.read_pickle(os.path.join(STRUCTURAL_DATA_DIR, "MICrONS_list_simplices_by_dimension%s.pkl" % str))["original"]


def load_functional_data(npzf_name, conn_mat):
    """Loads ''spikes'' and oracle score from MICrONS functional data (saved to npz in `assemblyfire`),
    maps idx to the structural connectome, and padds oracle score it with nans to have the same shape
    as the structural data"""
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
    spikes = spikes[idx_tmp[sort_idx], :]
    # padd oracle scores with NaNs
    idx = conn_mat.vertices.index.to_numpy()
    data = np.full(len(idx), np.nan)
    data[oracle_scores.index.to_numpy()] = oracle_scores.to_numpy()
    return spikes, gids[sort_idx], pd.Series(data, index=idx)


def _population_coupling(binned_spikes):
    """Fast implementation of population coupling (aka. coupling coefficient: corr of binned spikes
    with the avg. of the binned spikes of the rest of neurons') by Michael"""
    binned_spikes = binned_spikes - binned_spikes.mean(axis=1, keepdims=True)
    mn_pop = binned_spikes.mean(axis=0)
    A = np.dot(binned_spikes, mn_pop.reshape((-1, 1)))[:, 0]
    B = np.sqrt(np.var(mn_pop) * np.var(binned_spikes, axis=1)) * binned_spikes.shape[1]
    return A / B


def population_coupling(binned_spikes, n_ctrls=10, seed=12345):
    """Z-scored version of `_population_coupling()` above"""
    ccs = _population_coupling(binned_spikes)
    cc_ctrls = np.zeros((n_ctrls, binned_spikes.shape[0]), dtype=np.float32)
    for i in range(n_ctrls):
        np.random.seed(seed + i)
        shuffled_binned_spikes = binned_spikes.copy()
        np.random.shuffle(shuffled_binned_spikes)  # shuffle's only rows
        np.random.shuffle(shuffled_binned_spikes.T)  # transpose and shuffle rows -> shuffle columns
        cc_ctrls[i, :] = _population_coupling(shuffled_binned_spikes)
    means, stds = np.mean(cc_ctrls, axis=0), np.std(cc_ctrls, axis=0)
    return (ccs - means) / stds


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
    sns.boxplot(x="layer", y="'rate'", order=[23, 4, 5, 6], linewidth=0.5, fliersize=0, data=df, ax=ax)
    sns.stripplot(x="layer", y="'rate'", order=[23, 4, 5, 6], color="black", dodge=True, size=1., jitter=0.1, data=df, ax=ax)
    sns.despine(trim=True, offset=1)
    fig.savefig(fig_name, bbox_inches="tight", transparent=True)
    plt.close(fig)


def plot_oracle_score_vs_x(df, x, fig_name):
    """Scatter plot (and regression line) of anything (e.g. rate or CC) vs. oracle score"""
    fig = plt.figure(figsize=(1.8, 1.6))
    ax = fig.add_subplot(1, 1, 1)
    sns.regplot(x=x, y="oracle_score", marker='.', scatter_kws={"s": 7, "edgecolor": "none"},
                line_kws={"linewidth": 0.5}, data=df, ax=ax)
    ax.text(mean_df[x].quantile(0.99), 0, "r = %.2f" % pearsonr(df[x].to_numpy(), df["oracle_score"].to_numpy()).statistic,
            horizontalalignment="center", fontsize=5)
    sns.despine(trim=True, offset=1)
    fig.savefig(fig_name, bbox_inches="tight", transparent=True)
    plt.close(fig)


def plot_oracle_scores_vs_sdim(stats, node_part_sums, fig_name, text_offset=0.01):
    """Plot oracle score vs. simplex dimension (and position)"""
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
    ax.set_ylabel("Mean oracle score")
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


def main(conn_mat, session_idx, scan_idx, maximal=False, only_l234=False, plt_rates=False):
    max_str = "_max_" if maximal else "_"
    l234_str = "_L234_" if only_l234 else "_"
    simplices = load_simplex_list(maximal=maximal)
    node_part = topology.node_participation(conn_mat.matrix, max_simplices=maximal, threads=8)
    if only_l234:
        node_part.loc[conn_mat.vertices["cell_type"].isin(L56_MTYPES), :] = 0
    all_node_part_sums = node_part.sum()

    sum_stats, frac_node_parts, dfs = {}, {}, []
    for session_id, scan_id in zip(session_idx, scan_idx):
        name_tag = "MICrONS_session%i_scan%i" % (session_id, scan_id)
        spikes, gids, functional_data = load_functional_data(os.path.join(FUNCTIONAL_DATA_DIR, "%s.npz" % name_tag), conn_mat)
        if plt_rates:
            rates, ccs = np.mean(spikes, axis=1), population_coupling(spikes)
            tmp = functional_data.loc[functional_data.notna()]
            data = np.vstack([tmp.to_numpy(), rates, ccs]).T
            df = pd.DataFrame(data=data, columns=["oracle_score", "'rate'", "CC"], index=gids)
            df["mtype"] = conn_mat.vertices.loc[df.index, "cell_type"]
            df["layer"] = df["mtype"].map(LAYER_DICT)
            dfs.append(df)
            plot_lw_rates(df, "figs/%s_l-w_rate.pdf" % name_tag)
            plot_oracle_score_vs_x(df, "'rate'", "figs/%s_oracle_score_vs_rate.pdf" % name_tag)
            plot_oracle_score_vs_x(df, "CC", "figs/%s_oracle_score_vs_CC.pdf" % name_tag)
        if only_l234:
            mtypes = conn_mat.vertices.loc[functional_data.loc[functional_data.notna()].index, "cell_type"]
            functional_data.loc[mtypes.loc[mtypes.isin(L56_MTYPES)].index] = np.nan
        stats = agg_along_dims(nstats.node_stats_per_position(simplices, functional_data,
                                                              dims=simplices.index.drop(0), with_multiplicity=True))
        sum_stats[name_tag[8:]] = stats["all"]["mean"].to_numpy()
        node_part_sums = node_part.loc[functional_data.notna()].sum()
        frac_node_parts[name_tag[8:]] = node_part_sums.loc[1:].to_numpy() / all_node_part_sums.loc[1:].to_numpy()
        plot_oracle_scores_vs_sdim(stats, node_part_sums,
                                   "figs/%s_oracle_score%svs%ssimplex_dim.pdf" % (name_tag, l234_str, max_str))
    # summary plots across scans
    plot_y_vs_sdim_summary(sum_stats, "Mean oracle score",
                           "figs/MICrONS_oracle_score%svs%ssimplex_dim.pdf" % (l234_str, max_str))
    plot_y_vs_sdim_summary(frac_node_parts, "Node participation ratio",
                           "figs/MICrONS_node_part_sum%svs%ssimplex_dim.pdf" % (l234_str, max_str))
    if plt_rates:
        df = pd.concat(dfs)
        mean_df = df.groupby(df.index)[["oracle_score", "'rate'", "CC"]].agg("mean")
        mean_df["mtype"] = conn_mat.vertices.loc[mean_df.index, "cell_type"]
        mean_df["layer"] = mean_df["mtype"].map(LAYER_DICT)
        plot_lw_rates(df, "figs/MICrONS_l-w_rate.pdf")
        plot_oracle_score_vs_x(df, "'rate'", "figs/MICrONS_oracle_score_vs_rate.pdf")
        plot_oracle_score_vs_x(df, "CC", "figs/MICrONS_oracle_score_vs_CC.pdf")


if __name__ == "__main__":
    session_idx = [4, 5, 6, 6, 6, 7]  # , 8, 9],
    scan_idx = [7, 7, 2, 4, 7, 4]  # , 5, 3]
    conn_mat = load_connectome(STRUCTURAL_DATA_DIR, "MICrONS")
    main(conn_mat, session_idx, scan_idx, maximal=True, only_l234=True, plt_rates=True)
    main(conn_mat, session_idx, scan_idx, maximal=True, only_l234=False)
    main(conn_mat, session_idx, scan_idx, maximal=False, only_l234=True)
    main(conn_mat, session_idx, scan_idx, maximal=False, only_l234=False)







