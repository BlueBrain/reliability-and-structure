"""
Reliability in RC +/- vs. their controls
author: András Ecker, last update: 03.2024
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import kruskal
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
        "RC - 1 (ctrl)": "BlobStimReliability_O1v5-SONATA_RecipRemoval_Unstruct-0",
        "RC - 2": "BlobStimReliability_O1v5-SONATA_RecipRemoval_StructDim56_456",
        "RC - 2 (ctrl)": "BlobStimReliability_O1v5-SONATA_RecipRemoval_Unstruct-1",
        "RC - 3": "BlobStimReliability_O1v5-SONATA_RecipRemoval_StructDim456",
        "RC - 3 (ctrl)": "BlobStimReliability_O1v5-SONATA_RecipRemoval_Unstruct-2",
        "RC - 4": "BlobStimReliability_O1v5-SONATA_RecipRemoval_Unstruct-3",  # not possible to give it a pair
        "RC + 1": "BlobStimReliability_O1v5-SONATA_ConnAdd_RecipStruct0x2",
        "RC + 1 (ctrl)": "BlobStimReliability_O1v5-SONATA_ConnAdd_Control0x2",
        "RC + 2": "BlobStimReliability_O1v5-SONATA_ConnAdd_RecipStruct0x4",
        "RC + 2 (ctrl)": "BlobStimReliability_O1v5-SONATA_ConnAdd_Control0x4",
        "RC + 3": "BlobStimReliability_O1v5-SONATA_ConnAdd_RecipStruct0x8",
        "RC + 3 (ctrl)": "BlobStimReliability_O1v5-SONATA_ConnAdd_Control0x8",
        "RC + 4": "BlobStimReliability_O1v5-SONATA_ConnAdd_RecipStruct0x16",
        "RC + 4 (ctrl)": "BlobStimReliability_O1v5-SONATA_ConnAdd_Control0x16"}
COLS_REMOVE, COLS_ADD = plt.colormaps["summer_r"](np.linspace(0.2, 0.9, 4)), plt.cm.RdPu(np.linspace(0.2, 0.9, 4))
CMAP = {"RC - 1": COLS_REMOVE[0], "RC - 2": COLS_REMOVE[1],  "RC - 3": COLS_REMOVE[2], "RC - 4": COLS_REMOVE[3],
        "RC + 1": COLS_ADD[0], "RC + 2": COLS_ADD[1], "RC + 3": COLS_ADD[2], "RC + 4": COLS_ADD[3]}



def load_reliabilities():
    rels = {mod_name: np.load(os.path.join(SIM_DIR, sim_name, "working_dir", "reliability_basic.npz"))["reliability"]
            for mod_name, sim_name in SIMS.items()}
    return pd.DataFrame(rels)


def load_simplex_counts():
    scs = {mod_name: pd.read_pickle(os.path.join(SIM_DIR, sim_name,
                                                 "working_dir", "simplex_counts.pkl"))["simplex_counts_EXC"]
           for mod_name, sim_name in SIMS.items()}
    return pd.DataFrame(scs)


def plot_rels(df, x, fig_name):
    """KDE of reliabilities and their controls (+ baseline)"""
    fig = plt.figure(figsize=(1.8, 1.6))
    ax = fig.add_subplot(1, 1, 1)
    sns.kdeplot(x=x, color=CMAP[x], clip=[0, 1], data=df, ax=ax)
    if x != "RC - 4":
        sns.kdeplot(x=x + " (ctrl)", color="gray", ls="--", clip=[0, 1], data=df, ax=ax)
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
        ax.plot(df[x + " (ctrl)"], color="gray", ls="--")
    ax.plot(df["baseline"], color="black", ls="--", lw=0.75)
    ax.set_xticks([0, 5])
    ax.set_yticks([])
    sns.despine(trim=True, left=True, offset=1)
    fig.savefig(fig_name, bbox_inches="tight", transparent=True)
    plt.close(fig)


def plot_rel_means(rel_means, pvals, fig_name, sign_y=0.1):
    """Plots means relative to baseline"""
    colors, edge_colors, sign = [], [], []
    for i, mod_name in enumerate(rel_means.index.to_numpy()):
        if " (ctrl)" in mod_name:
            colors.append("white")
            edge_colors.append(CMAP[mod_name.replace(" (ctrl)", "")])
        else:
            colors.append(CMAP[mod_name])
            edge_colors.append(CMAP[mod_name])
        if pvals[mod_name] < 0.05:
            sign.append(i)

    fig = plt.figure(figsize=(4.5, 1.6))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(np.arange(len(rel_means)), rel_means, color=colors, edgecolor=edge_colors)
    ax.plot([-1, len(rel_means)], [0, 0], color="black", ls="dotted", lw=0.75)
    for x in sign:
        ax.text(x, sign_y, "*", ha="center", fontsize=9)
    ax.set_xticks([])
    sns.despine(trim=True, bottom=True, offset=2)
    fig.savefig(fig_name, bbox_inches="tight", transparent=True)
    plt.close(fig)


def plot_scatter(rel_edges, rel_means, fig_name):
    """Plots means vs. pct. of edges changed"""
    rem = ["RC - %i" %i for i in range(1, 5)]
    rem_ctrl = ["RC - %i (ctrl)" %i for i in range(1, 4)]
    add = ["RC + %i" %i for i in range(1, 5)]
    add_ctrl = ["RC + %i (ctrl)" % i for i in range(1, 5)]
    colors = []
    for mod_name in rel_means.index.to_numpy():
        if " (ctrl)" in mod_name:
            colors.append(CMAP[mod_name.replace(" (ctrl)", "")])
        else:
            colors.append(CMAP[mod_name])

    fig = plt.figure(figsize=(1., 0.8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(rel_edges[rem].to_numpy(), np.abs(rel_means[rem].to_numpy()), c=CMAP["RC - 2"])
    ax.plot(rel_edges[rem_ctrl].to_numpy(), np.abs(rel_means[rem_ctrl].to_numpy()), c=CMAP["RC - 2"], ls="--")
    ax.plot(rel_edges[add].to_numpy(), np.abs(rel_means[add].to_numpy()), c=CMAP["RC + 2"])
    ax.plot(rel_edges[add_ctrl].to_numpy(), np.abs(rel_means[add_ctrl].to_numpy()), c=CMAP["RC + 2"], ls="--")
    ax.scatter(rel_edges.to_numpy(), np.abs(rel_means.to_numpy()), s=10, c=colors)
    sns.despine(trim=True, offset=2)
    fig.savefig(fig_name, bbox_inches="tight", transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    rels = load_reliabilities()
    scs = load_simplex_counts()
    for x in ["RC - %i" % i for i in range(1, 5)] + ["RC + %i" % i for i in range(1, 5)]:
        plot_rels(rels, x, "figs/paper/%s_reliability.pdf" % x.replace(" ", ""))
        plot_scounts(scs, x, "figs/paper/%s_simplex_counts.pdf" % x.replace(" ", ""))
    means = rels.mean()
    rel_means = means.drop("baseline") - means["baseline"]
    pvals = {mod_name: kruskal(rels["baseline"].to_numpy(), rels[mod_name].to_numpy(), nan_policy="omit").pvalue
             for mod_name in rel_means.index.to_numpy()}
    plot_rel_means(rel_means, pvals, "figs/paper/rel_means.pdf")
    n_edges = scs.loc[1]
    rel_edges = np.abs(1 - n_edges.drop("baseline") / n_edges["baseline"]) * 100
    plot_scatter(rel_edges, rel_means, "figs/paper/rel_means_vs_rel_edges.pdf")










