"""
Plotting oracle score (kind of reliability) against simplex dim (and position within simplex) for MICrONS
author: András Ecker, last update: 02.2024
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, zscore
import sys
sys.path.append("../../../library")
from coupling_coefficient import normalize_cc
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
COLORS = {"all": "tab:orange", "sink": "tab:green", "source": "tab:blue"}
MARKERS = {"all": "o", "sink": ">", "source":"<"}


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
    fig.savefig(fig_name, bbox_inches="tight", dpi=100)
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
    fig.savefig(fig_name, bbox_inches="tight", dpi=100)
    plt.close(fig)


def plot_y_vs_sdim(stats, node_part_sums, ylabel, fig_name, text_offset=0.01):
    """Plot anything (e.g. a functional feature like oracle score) vs. simplex dimension (and position)"""
    fig = plt.figure(figsize=(1.8, 1.6))
    ax = fig.add_subplot(1, 1, 1)
    for pos, color in COLORS.items():
        x = stats[pos].index.to_numpy()
        ax.plot(x, stats[pos]["mean"], color=color, marker=MARKERS[pos], label=pos)
        ax.fill_between(x, stats[pos]["mean"] - stats[pos]["sem"], stats[pos]["mean"] + stats[pos]["sem"],
                        color=color, alpha=0.2)
    y = stats["all"]["mean"].max() + text_offset
    for x_ in x:
        ax.text(x_, y, node_part_sums[x_], horizontalalignment="center", rotation="vertical", fontsize=6)
    ax.set_xticks(x)
    ax.set_xlabel("Simplex dimension")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    sns.despine(trim=True, offset=1)
    fig.savefig(fig_name, bbox_inches="tight", dpi=100)
    plt.close(fig)


def plot_y_vs_sdim_paper(stats, ylim, fig_name, legend=True):
    """Plot anything (e.g. a functional feature like oracle score) vs. simplex dimension (and position)"""
    fig = plt.figure(figsize=(1., 0.9))
    ax = fig.add_subplot(1, 1, 1)
    for pos, color in COLORS.items():
        x = stats[pos].index.to_numpy()
        ax.plot(x, stats[pos]["mean"], color=color, marker=MARKERS[pos], markersize=2, lw=0.9, label=pos)
        ax.fill_between(x, stats[pos]["mean"] - stats[pos]["sem"], stats[pos]["mean"] + stats[pos]["sem"],
                        color=color, alpha=0.2)
    ax.set_xticks(np.arange(1, 8))  # pretty hard coded...
    ax.set_ylim(ylim)
    if legend:
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
    fig.savefig(fig_name, bbox_inches="tight", dpi=100)
    plt.close(fig)


def plot_y_vs_sdim_summary_paper(dict, ylim, fig_name):
    """Plot anything (e.g. oracle score or node part sums) vs. simplex dimension across scans and sessions"""
    fig = plt.figure(figsize=(1., 0.9))
    ax = fig.add_subplot(1, 1, 1)
    x = np.arange(1, 8)  # pretty hard coded...
    ys = np.full((len(dict), len(x)), np.nan)
    for i, (label, y) in enumerate(dict.items()):
        ax.plot(x, y, marker="o", markersize=2, lw=0.9, label=label)
        ys[i, :] = y
    ax.plot(x, np.nanmean(ys, axis=0), color="black", marker="o", markersize=2, lw=0.9, label="mean")
    ax.set_xticks(x)
    ax.set_ylim(ylim)
    ax.legend(frameon=False, bbox_to_anchor=(1., 1.1))
    sns.despine(trim=True, offset=1)
    fig.savefig(fig_name, bbox_inches="tight", transparent=True)
    plt.close(fig)


def main_functional(conn_mat, fn_df):
    dfs = []
    for session in fn_df.columns.get_level_values(0).unique():
        df = fn_df[session]
        df = df.loc[df["rate"].notna()]
        df["mtype"] = conn_mat.vertices.loc[df.index, "cell_type"]
        df["layer"] = df["mtype"].map(LAYER_DICT)
        dfs.append(df)
        plot_lw_rates(df, "figs/MICrONS_%s_l-w_rate.pdf" % session)
        plot_oracle_score_vs_x(df, "rate", "figs/MICrONS_%s_oracle_score_vs_rate.pdf" % session)
        plot_oracle_score_vs_x(df, "coupling_coeff", "figs/MICrONS_%s_oracle_score_vs_coupling_coeff.pdf" % session)
    df = pd.concat(dfs)
    mean_df = df.groupby(df.index)[["rate", "oracle_score", "coupling_coeff"]].agg("mean")
    mean_df["mtype"] = conn_mat.vertices.loc[mean_df.index, "cell_type"]
    mean_df["layer"] = mean_df["mtype"].map(LAYER_DICT)
    plot_lw_rates(df, "figs/MICrONS_l-w_rate.pdf")
    plot_oracle_score_vs_x(df, "rate", "figs/MICrONS_oracle_score_vs_rate.pdf")
    plot_oracle_score_vs_x(df, "coupling_coeff", "figs/MICrONS_oracle_score_vs_coupling_coeff.pdf")


def main(conn_mat, fn_df, fn_feature="oracle_score", maximal=False, only_l234=False):
    max_str = "_max_" if maximal else "_"
    l234_str = "_L234_" if only_l234 else "_"
    simplices = load_simplex_list(maximal=maximal)
    node_part = topology.node_participation(conn_mat.matrix, max_simplices=maximal, threads=8)
    if only_l234:
        node_part.loc[~conn_mat.vertices["cell_type"].isin(L234_MTYPES), :] = 0
    all_node_part_sums = node_part.sum()

    sum_stats, frac_node_parts = {}, {}
    for session in fn_df.columns.get_level_values(0).unique():
        if fn_feature != "coupling_coeff":
            functional_data = fn_df[session, fn_feature].copy()
        else:
            functional_data = normalize_cc(fn_df[session].drop(columns=["rate", "oracle_score"]))
        if only_l234:
            mtypes = conn_mat.vertices.loc[functional_data.loc[functional_data.notna()].index, "cell_type"]
            functional_data.loc[mtypes.loc[~mtypes.isin(L234_MTYPES)].index] = np.nan
        stats = agg_along_dims(nstats.node_stats_per_position(simplices, functional_data,
                                                              dims=simplices.index.drop(0), with_multiplicity=True))
        sum_stats[session] = stats["all"]["mean"].to_numpy()
        node_part_sums = node_part.loc[functional_data.notna()].sum()
        frac_node_parts[session] = node_part_sums.loc[1:].to_numpy() / all_node_part_sums.loc[1:].to_numpy()
        plot_y_vs_sdim(stats, node_part_sums, "Mean %s" % fn_feature,
                       "figs/MICrONS_%s_%s%svs%ssimplex_dim.png" % (session, fn_feature, l234_str, max_str))
        if maximal and not only_l234:
            legend = True if session_id == 4 and scan_id == 7 else False
            ylim = [0.05, 0.35] if fn_feature == "oracle_score" else [-1.5, 1.5]
            plot_y_vs_sdim_paper(stats, ylim, "figs/paper/MICrONS_%s_%s.pdf" % (session, fn_feature), legend=legend)
    # summary plot (using raw data from all scans)
    if fn_feature != "coupling_coeff":
        functional_data = fn_df.loc[:, fn_df.columns.get_level_values(1) == fn_feature].copy()
        functional_data = zscore(functional_data, nan_policy="omit").mean(axis=1)  # zscore columns and take their mean
    else:
        functional_data = normalize_cc(fn_df)
    if only_l234:
        mtypes = conn_mat.vertices.loc[functional_data.loc[functional_data.notna()].index, "cell_type"]
        functional_data.loc[mtypes.loc[~mtypes.isin(L234_MTYPES)].index] = np.nan
    stats = agg_along_dims(nstats.node_stats_per_position(simplices, functional_data,
                                                          dims=simplices.index.drop(0), with_multiplicity=True))
    node_part_sums = node_part.loc[functional_data.notna()].sum()
    plot_y_vs_sdim(stats, node_part_sums, "Mean %s" % fn_feature,
                   "figs/MICrONS_all_scans_%s%svs%ssimplex_dim.png" % (fn_feature, l234_str, max_str))
    # summary plots (using results from all scans)
    plot_y_vs_sdim_summary(sum_stats, "Mean %s" % fn_feature,
                           "figs/MICrONS_%s%svs%ssimplex_dim.png" % (fn_feature, l234_str, max_str))
    if maximal and not only_l234:
        plot_y_vs_sdim_summary_paper(sum_stats, ylim, "figs/paper/MICrONS_%s.pdf" % fn_feature)
    plot_y_vs_sdim_summary(frac_node_parts, "Node participation ratio",
                           "figs/MICrONS_node_part_sum%svs%ssimplex_dim.png" % (l234_str, max_str))


if __name__ == "__main__":
    conn_mat = load_connectome(STRUCTURAL_DATA_DIR, "MICrONS")
    fn_df = pd.read_pickle(os.path.join(FUNCTIONAL_DATA_DIR, "MICrONS_functional_summary.pkl"))
    main_functional(conn_mat, fn_df)
    main(conn_mat, fn_df, maximal=True, only_l234=True)
    main(conn_mat, fn_df, maximal=True, only_l234=False)
    main(conn_mat, fn_df, maximal=False, only_l234=True)
    main(conn_mat, fn_df, maximal=False, only_l234=False)
    main(conn_mat, fn_df, fn_feature="coupling_coeff", maximal=True, only_l234=True)
    main(conn_mat, fn_df, fn_feature="coupling_coeff", maximal=True, only_l234=False)
    main(conn_mat, fn_df, fn_feature="coupling_coeff", maximal=False, only_l234=True)
    main(conn_mat, fn_df, fn_feature="coupling_coeff", maximal=False, only_l234=False)







