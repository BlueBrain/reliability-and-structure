"""
Reliability in RC +/- vs. their controls
author: András Ecker, last update: 03.2024
"""

import os
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import kruskal
from conntility import ConnectivityMatrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="ticks", context="paper", font="Helvetica Neue",
        rc={"axes.labelsize": 7, "legend.fontsize": 6, "axes.linewidth": 0.6, "xtick.labelsize": 6, "ytick.labelsize": 6,
            "xtick.major.size": 2, "xtick.major.width": 0.5, "xtick.minor.size": 1.5, "xtick.minor.width": 0.3,
            "ytick.major.size": 2, "ytick.major.width": 0.5, "ytick.minor.size": 1.5, "ytick.minor.width": 0.3})
SIM_DIR = "/gpfs/bbp.cscs.ch/data/scratch/proj9/bisimplices/simulations/"
SIMS = {"baseline": "BlobStimReliability_O1v5-SONATA_Baseline",
        "RC - 1": "BlobStimReliability_O1v5-SONATA_RecipRemoval_StructDim56",
        "RC* - 1": "BlobStimReliability_O1v5-SONATA_RecipRemoval_Unstruct-0",
        "RC - 2": "BlobStimReliability_O1v5-SONATA_RecipRemoval_StructDim56_456",
        "RC* - 2": "BlobStimReliability_O1v5-SONATA_RecipRemoval_Unstruct-1",
        "RC - 3": "BlobStimReliability_O1v5-SONATA_RecipRemoval_StructDim456",
        "RC* - 3": "BlobStimReliability_O1v5-SONATA_RecipRemoval_Unstruct-2",
        "RC - 4": "BlobStimReliability_O1v5-SONATA_RecipRemoval_Unstruct-3",  # not possible to give it a pair
        "RC + 1": "BlobStimReliability_O1v5-SONATA_ConnAdd_RecipStruct0x2",
        "RC* + 1": "BlobStimReliability_O1v5-SONATA_ConnAdd_Control0x2",
        "RC + 2": "BlobStimReliability_O1v5-SONATA_ConnAdd_RecipStruct0x3",
        "RC* + 2": "BlobStimReliability_O1v5-SONATA_ConnAdd_Control0x3",
        "RC + 3": "BlobStimReliability_O1v5-SONATA_ConnAdd_RecipStruct0x4",
        "RC* + 3": "BlobStimReliability_O1v5-SONATA_ConnAdd_Control0x4",
        "RC + 4": "BlobStimReliability_O1v5-SONATA_ConnAdd_RecipStruct0x5",
        "RC* + 4": "BlobStimReliability_O1v5-SONATA_ConnAdd_Control0x5",
        "RC + 5": "BlobStimReliability_O1v5-SONATA_ConnAdd_RecipStruct0x8",
        "RC* + 5": "BlobStimReliability_O1v5-SONATA_ConnAdd_Control0x8"} | {"%ik" % i:
        "BlobStimReliability_O1v5-SONATA_ConnRewireEnhanced%iK" % i for i in [100, 200, 300, 400, 500, 670]}
        # "RC + 6": "BlobStimReliability_O1v5-SONATA_ConnAdd_RecipStruct0x16",
        # "RC* + 6": "BlobStimReliability_O1v5-SONATA_ConnAdd_Control0x16"}
COLS_REMOVE, COLS_ENH = plt.colormaps["summer_r"](np.linspace(0.2, 0.9, 4)), plt.cm.Reds(np.linspace(0.2, 0.9, 6))
COLS_ADD = plt.cm.RdPu(np.linspace(0.2, 0.9, 4))
CMAP = {"RC - 1": COLS_REMOVE[3], "RC - 2": COLS_REMOVE[2],  "RC - 3": COLS_REMOVE[1], "RC - 4": COLS_REMOVE[0],
        "100k": COLS_ENH[5], "200k": COLS_ENH[4], "300k": COLS_ENH[3], "400k": COLS_ENH[2], "500k": COLS_ENH[1],
        "670k": COLS_ENH[0], "RC + 1": COLS_ADD[3], "RC + 2": COLS_ADD[2], "RC + 3": COLS_ADD[1], "RC + 4": COLS_ADD[0],
        "RC + 5": "lightsalmon"}


def load_reliabilities():
    return pd.DataFrame({mod_name: np.load(os.path.join(SIM_DIR, sim_name, "working_dir",
                                                        "reliability_basic.npz"))["reliability"]
                         for mod_name, sim_name in SIMS.items()})


def load_simplex_counts():
    dfs = {}
    for mod_name, sim_name in SIMS.items():
        if mod_name[-1] != "k":
            dfs[mod_name] = pd.read_pickle(os.path.join(SIM_DIR, sim_name, "working_dir", "simplex_counts.pkl"))["simplex_counts_EXC"]
        else:
            dfs[mod_name] = pd.read_pickle(os.path.join(SIM_DIR, sim_name, "working_dir", "simplex_counts_EE.pkl"))
    return pd.DataFrame(dfs)


def get_edges_diff():
    """Gets difference in edges from saved connectivity matrices"""
    conn_mats = {mod_name: ConnectivityMatrix.from_h5(os.path.join(SIM_DIR, sim_name, "working_dir",
                                              "connectome.h5")).index("synapse_class").isin("EXC").matrix.astype(int)
                 for mod_name, sim_name in SIMS.items()}
    rel_edges = pd.Series({mod_name: (conn_mat - conn_mats["baseline"]).nnz / 2 / conn_mats["baseline"].nnz * 100
                          for mod_name, conn_mat in conn_mats.items()})
    return rel_edges.drop("baseline")


def plot_rels(df, x, fig_name):
    """KDE of reliabilities and their controls (+ baseline)"""
    fig = plt.figure(figsize=(1.8, 1.6))
    ax = fig.add_subplot(1, 1, 1)
    sns.kdeplot(x=x, color=CMAP[x], clip=[0, 1], data=df, ax=ax)
    if x != "RC - 4":
        sns.kdeplot(x=x.replace("RC", "RC*"), color="gray", ls="--", clip=[0, 1], data=df, ax=ax)
    sns.kdeplot(x="baseline", color="black", ls="--", lw=0.75, alpha=0.75, clip=[0, 1], data=df, ax=ax)
    ax.set_xlim([0, 1])
    ax.set_xlabel("")
    ax.set_ylabel("")
    sns.despine(trim=True, offset=2)
    fig.savefig(fig_name, bbox_inches="tight", transparent=True)
    plt.close(fig)


def plot_scounts(df, x, fig_name):
    """Plots simplex counts and their controls (+ baseline)"""
    fig = plt.figure(figsize=(1., 0.8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(df[x], color=CMAP[x])
    if x != "RC - 4":
        ax.plot(df[x.replace("RC", "RC*")], color="gray", ls="--")
    ax.plot(df["baseline"], color="black", ls="--", lw=0.75)
    ax.set_xticks([0, 5])
    ax.set_yticks([])
    sns.despine(trim=True, left=True, offset=1)
    fig.savefig(fig_name, bbox_inches="tight", transparent=True)
    plt.close(fig)


def plot_pvals(pvals, fig_name):
    """Plot p-value matrix"""
    fig = plt.figure(figsize=(1.8, 1.6))
    ax = fig.add_subplot(1, 1, 1)
    i = ax.imshow(pvals, cmap="viridis", interpolation="nearest", aspect="auto")
    fig.colorbar(i)
    ax.set_xticks(np.arange(len(pvals.columns)))
    ax.set_xticklabels(pvals.columns.to_numpy(), fontsize=5, rotation=90)
    ax.set_yticks(np.arange(len(pvals.index)))
    ax.set_yticklabels(pvals.index.to_numpy(), fontsize=5)
    fig.savefig(fig_name, bbox_inches="tight", transparent=True)
    plt.close(fig)


def plot_rel_means(rel_means, pvals, fig_name, sign_y=0.07):
    """Plots means relative to baseline"""
    colors, hatches, sign = [], [], []
    for i, mod_name in enumerate(rel_means.index.to_numpy()):
        if "*" in mod_name:
            colors.append(CMAP[mod_name.replace(" (ctrl)", "")])
            hatches.append('//')
        else:
            colors.append(CMAP[mod_name])
            if mod_name != "RC - 4":
                hatches.append('oo')
            else:
                hatches.append('oo//')
        if pvals.loc["baseline", mod_name] < 0.05:
            sign.append(i)

    fig = plt.figure(figsize=(1.8, 1.6))
    ax = fig.add_subplot(1, 1, 1)
    bars = ax.bar(np.arange(len(rel_means)), rel_means, color="white", edgecolor=colors)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax.plot([-1, len(rel_means)], [0, 0], color="black", ls="dotted", lw=0.75)
    for x in sign:
        ax.text(x, sign_y, "*", ha="center", fontsize=9)
    ax.set_xticks([])
    sns.despine(trim=True, bottom=True, offset=2)
    fig.savefig(fig_name, bbox_inches="tight", transparent=True)
    plt.close(fig)


def plot_scatter(rel_edges, rel_means, fig_name):
    """Plots means vs. pct. of edges changed"""
    colors, mod, mod_ctrl, enh = [], [], [], []
    for mod_name in rel_means.index.to_numpy():
        if "*" in mod_name:
            colors.append(CMAP[mod_name.replace("RC*", "RC")])
            mod_ctrl.append(mod_name)
        elif mod_name[-1] == "k":
            colors.append(CMAP[mod_name])
            enh.append(mod_name)
        else:
            colors.append(CMAP[mod_name])
            mod.append(mod_name)

    fig = plt.figure(figsize=(1., 0.8))
    ax = fig.add_subplot(1, 1, 1)
    if len(enh):
        ax.plot(rel_edges[mod].to_numpy(), rel_means[mod].to_numpy(), c=CMAP["RC + 3"])
        ax.plot(rel_edges[enh].to_numpy(), rel_means[enh].to_numpy(), c=CMAP["300k"], ls="--")
    else:
        ax.plot(rel_edges[mod].to_numpy(), rel_means[mod].to_numpy(), c=CMAP["RC - 2"])
    ax.plot(rel_edges[mod_ctrl].to_numpy(), rel_means[mod_ctrl].to_numpy(), c="gray", ls="--")
    ax.scatter(rel_edges.to_numpy(), rel_means.to_numpy(), s=10, c=colors)
    sns.despine(trim=True, offset=1)
    fig.savefig(fig_name, bbox_inches="tight", transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    rels = load_reliabilities()
    scs = load_simplex_counts()
    for x in ["RC - %i" % i for i in range(1, 5)] + ["RC + %i" % i for i in range(1, 6)]:
        plot_rels(rels, x, "figs/paper/%s_reliability.pdf" % x.replace(" ", ""))
        plot_scounts(scs, x, "figs/paper/%s_simplex_counts.pdf" % x.replace(" ", ""))

    mod_names = rels.columns
    data = np.full((len(mod_names), len(mod_names)), np.nan, dtype=np.float32)
    row_idx, col_idx = np.triu_indices(len(mod_names), k=1)
    for i, j in zip(row_idx, col_idx):
        data[i, j] = kruskal(rels[mod_names[i]], rels[mod_names[j]], nan_policy="omit").pvalue
    pvals = pd.DataFrame(data, index=mod_names, columns=mod_names)
    plot_pvals(pvals.iloc[1:18, 1:18], "figs/paper/rel_pvals.pdf")
    means = rels.mean()
    rel_means = means.drop("baseline") - means["baseline"]
    plot_rel_means(rel_means.iloc[:7], pvals, "figs/paper/rel_means-.pdf", sign_y=0.0001)
    plot_rel_means(rel_means.iloc[7:17], pvals, "figs/paper/rel_means+.pdf")
    rel_edges = get_edges_diff()
    plot_scatter(rel_edges.iloc[:7], rel_means.iloc[:7], "figs/paper/rel_means_vs_rel_edges-.pdf")
    plot_scatter(rel_edges.iloc[7:], rel_means.iloc[7:], "figs/paper/rel_means_vs_rel_edges+.pdf")










