"""
Plotting oracle score (kind of reliability) against simplex dim (and position within simplex) for MICrONS
author: András Ecker, last update: 01.2024
"""

import os
import numpy as np
import pandas as pd
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
    oracle_scores = pd.Series(tmp["oracle_scores"], index=idx, name="oracle score")
    # match idx to structural data
    valid_idx = conn_mat.vertices.id[np.isin(conn_mat.vertices.id, oracle_scores.index)]
    oracle_scores = oracle_scores.loc[oracle_scores.index.isin(valid_idx)]
    valid_idx_tmp = pd.Series(valid_idx.index.to_numpy(), index=valid_idx.to_numpy())
    oracle_scores = pd.Series(oracle_scores.to_numpy(), index=valid_idx_tmp.loc[oracle_scores.index].to_numpy(),
                              name="oracle score").sort_index()
    # index out spikes as well (and sort gids)
    idx_tmp = np.where(np.in1d(idx, valid_idx_tmp.index.to_numpy()))[0]
    gids = valid_idx_tmp.loc[idx[idx_tmp]].to_numpy()
    sort_idx = np.argsort(gids)
    spikes = spikes[idx_tmp[sort_idx], :]
    # padd oracle scores with NaNs
    idx = conn_mat.vertices.index.to_numpy()
    data = np.full(len(idx), np.nan)
    data[oracle_scores.index.to_numpy()] = oracle_scores.to_numpy()
    return spikes, gids[sort_idx], pd.Series(data, index=idx)


def population_coupling(binned_spikes):
    """Fast implementation of population coupling (aka. coupling coefficient: corr of binned spikes
    with the avg. of the binned spikes of the rest of neurons') by Michael"""
    binned_spikes = binned_spikes - binned_spikes.mean(axis=1, keepdims=True)
    mn_pop = binned_spikes.mean(axis=0)
    A = np.dot(binned_spikes, mn_pop.reshape((-1, 1)))[:, 0]
    B = np.sqrt(np.var(mn_pop) * np.var(binned_spikes, axis=1)) * binned_spikes.shape[1]
    return A / B


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


def main(conn_mat, session_idx, scan_idx, maximal=False, only_l234=False, n_ctrl=10):
    max_str = "_max_" if maximal else "_"
    l234_str = "_L234_" if only_l234 else "_"
    simplices = load_simplex_list(maximal=maximal)
    node_part = topology.node_participation(conn_mat.matrix, max_simplices=maximal, threads=8)
    if only_l234:
        node_part.loc[conn_mat.vertices["cell_type"].isin(L56_MTYPES), :] = 0
    all_node_part_sums = node_part.sum()

    sum_stats, frac_node_parts = {}, {}
    for session_id, scan_id in zip(session_idx, scan_idx):
        name_tag = "MICrONS_session%i_scan%i" % (session_id, scan_id)
        spikes, gids, functional_data = load_functional_data(os.path.join(FUNCTIONAL_DATA_DIR, "%s.npz" % name_tag), conn_mat)
        cc = population_coupling(spikes)
        # TODO: cont. from here
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
    plot_y_vs_sdim_summary(sum_stats, "Mean oracle score",
                           "figs/MICrONS_oracle_score%svs%ssimplex_dim.pdf" % (l234_str, max_str))
    plot_y_vs_sdim_summary(frac_node_parts, "Node participation ratio",
                           "figs/MICrONS_node_part_sum%svs%ssimplex_dim.pdf" % (l234_str, max_str))


if __name__ == "__main__":
    # sessions 8, 5 are the ones that we usually discard
    session_idx, scan_idx = [4, 5, 6, 6, 6, 7, 8, 9], [7, 7, 2, 4, 7, 4, 5, 3]
    conn_mat = load_connectome(STRUCTURAL_DATA_DIR, "MICrONS")
    main(conn_mat, session_idx, scan_idx, maximal=True, only_l234=True)
    main(conn_mat, session_idx, scan_idx, maximal=True, only_l234=False)
    main(conn_mat, session_idx, scan_idx, maximal=False, only_l234=True)
    main(conn_mat, session_idx, scan_idx, maximal=False, only_l234=False)







