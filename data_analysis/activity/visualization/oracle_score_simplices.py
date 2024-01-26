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
COLORS = {"all": "tab:orange", "sink": "tab:green", "source": "tab:blue"}
MARKERS = {"all": "o", "sink": ">", "source":"<"}


def load_simplex_list(maximal=True):
    """Loads saved simplex list (for MICrONS)"""
    str = "_maximal" if maximal else "_"
    return pd.read_pickle(os.path.join(STRUCTURAL_DATA_DIR, "MICrONS_list_simplices_by_dimension%s.pkl" % str))["original"]


def load_functional_data(npzf_name, conn_mat):
    """Loads oracle score from MICrONS functional data (saved to npz in `assemblyfire`),
    maps idx to the structural connectome, and padds it with nans to have the same shape as the structural data"""
    tmp = np.load(npzf_name)
    oracle_scores = pd.Series(tmp["oracle_scores"], index=tmp["idx"], name="oracle score")
    # match idx to structural data
    valid_idx = conn_mat.vertices.id[np.isin(conn_mat.vertices.id, oracle_scores.index)]
    oracle_scores = oracle_scores.loc[oracle_scores.index.isin(valid_idx)]
    valid_idx_tmp = pd.Series(valid_idx.index.to_numpy(), index=valid_idx.to_numpy())
    oracle_scores = pd.Series(oracle_scores.to_numpy(), index=valid_idx_tmp.loc[oracle_scores.index].to_numpy(),
                              name="oracle score").sort_index()
    # padd with NaNs
    idx = conn_mat.vertices.index.to_numpy()
    data = np.full(len(idx), np.nan)
    data[oracle_scores.index.to_numpy()] = oracle_scores.to_numpy()
    return pd.Series(data, index=idx)


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


def plot_oracle_score_vs_sdim_summary(sum_stats, fig_name):
    """Plot oracle score vs. simplex dimension across scans and sessions"""
    fig = plt.figure(figsize=(1.8, 1.6))
    ax = fig.add_subplot(1, 1, 1)
    x = np.arange(1, 8)  # pretty hard coded...
    for label, y in sum_stats.items():
        ax.plot(x, y, label=label)
    ax.set_xticks(x)
    ax.set_xlabel("Simplex dimension")
    ax.set_ylabel("Mean oracle score")
    ax.legend(frameon=False, bbox_to_anchor=(1., 1.1))
    sns.despine(trim=True, offset=1)
    fig.savefig(fig_name, bbox_inches="tight", transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    conn_mat = load_connectome(STRUCTURAL_DATA_DIR, "MICrONS")
    for maximal, plt_str in zip([True, False], ["_max_", "_"]):
        simplices = load_simplex_list(maximal=maximal)
        node_part = topology.node_participation(conn_mat.matrix, max_simplices=maximal, threads=8)
        sum_stats = {}
        for session_id, scan_id in zip([4, 5, 6, 6, 6, 7], [7, 7, 2, 4, 7, 4]):
            name_tag = "MICrONS_session%i_scan%i" % (session_id, scan_id)
            functional_data = load_functional_data(os.path.join(FUNCTIONAL_DATA_DIR, "%s.npz" % name_tag), conn_mat)
            stats = agg_along_dims(nstats.node_stats_per_position(simplices, functional_data,
                                                                  dims=simplices.index.drop(0), with_multiplicity=True))
            sum_stats[name_tag[8:]] = stats["all"]["mean"].to_numpy()
            node_part_sums = node_part.loc[functional_data.notna()].sum()
            plot_oracle_scores_vs_sdim(stats, node_part_sums,
                                       "figs/%s_oracle_score_vs%ssimplex_dim.pdf" % (name_tag, plt_str))
        plot_oracle_score_vs_sdim_summary(sum_stats, "figs/MICrONS_oracle_score_vs%ssimplex_dim.pdf" % plt_str)





