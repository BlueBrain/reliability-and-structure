"""
Global vs. per-cell normalization of raw CC values
author: András Ecker, last update: 02.2024
"""

import os
import numpy as np
import sys
sys.path.append("../../../library")
from structural_basic import *
from utils_microns_bbp import add_firing_rates, add_cc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="ticks", context="paper", font="Helvetica Neue",
        rc={"axes.labelsize": 7, "legend.fontsize": 6, "axes.linewidth": 0.6, "xtick.labelsize": 6, "ytick.labelsize": 6,
            "xtick.major.size": 2, "xtick.major.width": 0.5, "xtick.minor.size": 1.5, "xtick.minor.width": 0.3,
            "ytick.major.size": 2, "ytick.major.width": 0.5, "ytick.minor.size": 1.5, "ytick.minor.width": 0.3})
STRUCTURAL_DATA_DIR = "/gpfs/bbp.cscs.ch/home/egassant/reliability_and_structure/data_analysis/data"
MICRONS_FN_DATA_DIR = "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/assemblyfire/MICrONS"
BBP_FN_DATA_DIR = "/gpfs/bbp.cscs.ch/data/scratch/proj9/bisimplices/bbp_workflow/7b381e96-91ac-4ddd-887b-1f563872bd1c"


def plot_norm_dists(df, name_tag, legend=True):
    """Plots different normalization of CC values and their correlation (colored with rate)"""
    fig = plt.figure(figsize=(1.8, 1.6))
    ax = fig.add_subplot(1, 1, 1)
    plt.hist(df["CC"].to_numpy(), bins=50, density=True, histtype="step", label="global")
    plt.hist(df["CC_norm_cell"].to_numpy(), bins=50, density=True, histtype="step", label="per-cell")
    if legend:
        ax.legend(frameon=False)
    sns.despine(trim=True, offset=1)
    fig.savefig("figs/paper/%s_CC_dists.pdf" % name_tag, bbox_inches="tight", transparent=True)
    plt.close(fig)

    fig = plt.figure(figsize=(1.8, 1.6))
    ax = fig.add_subplot(1, 1, 1)
    sc = ax.scatter(df["CC"].to_numpy(), df["CC_norm_cell"].to_numpy(), c=df["rates"].to_numpy(), cmap="viridis",
                    marker=".", s=3, edgecolor="none")
    fig.colorbar(sc)
    sns.despine(trim=True, offset=1)
    fig.savefig("figs/paper/%s_CC_corrs.png" % name_tag, bbox_inches="tight", transparent=True, dpi=500)
    plt.close(fig)


if __name__ == "__main__":
    conn_mat = load_connectome(STRUCTURAL_DATA_DIR, "MICrONS")
    conn_mat = add_firing_rates(conn_mat, "MICrONS", os.path.join(MICRONS_FN_DATA_DIR, "MICrONS_functional_summary.pkl"))
    conn_mat = add_cc(conn_mat, os.path.join(MICRONS_FN_DATA_DIR, "MICrONS_functional_summary.pkl"), "global")
    conn_mat = add_cc(conn_mat, os.path.join(MICRONS_FN_DATA_DIR, "MICrONS_functional_summary.pkl"), "per_cell")
    df = conn_mat.vertices[["rates", "CC", "CC_norm_cell"]]
    df = df.loc[df["rates"].notna()]
    plot_norm_dists(df, "MICrONS")

    conn_mat = load_connectome(STRUCTURAL_DATA_DIR, "BBP")
    conn_mat = add_firing_rates(conn_mat, "BBP", os.path.join(BBP_FN_DATA_DIR, "toposample_input",
                                                              "raw_spikes_exc.npy"), "toposample")
    conn_mat = add_cc(conn_mat, os.path.join(BBP_FN_DATA_DIR, "working_dir", "coupling_coefficients.pkl"), "global")
    conn_mat = add_cc(conn_mat, os.path.join(BBP_FN_DATA_DIR, "working_dir", "coupling_coefficients.pkl"), "per_cell")
    df = conn_mat.vertices[["rates", "CC", "CC_norm_cell"]]
    df = df.loc[df["rates"].notna()]
    plot_norm_dists(df, "BBP", legend=False)









