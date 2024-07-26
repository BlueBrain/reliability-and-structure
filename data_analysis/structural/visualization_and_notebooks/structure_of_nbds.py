# SPDX-FileCopyrightText: 2024 Blue Brain Project / EPFL
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Plotting basic features of neighbourhoods
author: András Ecker, last update: 02.2024
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="ticks", context="paper", font="Helvetica Neue",
        rc={"axes.labelsize": 7, "legend.fontsize": 6, "axes.linewidth": 0.6, "xtick.labelsize": 6, "ytick.labelsize": 6,
            "xtick.major.size": 2, "xtick.major.width": 0.5, "xtick.minor.size": 1.5, "xtick.minor.width": 0.3,
            "ytick.major.size": 2, "ytick.major.width": 0.5, "ytick.minor.size": 1.5, "ytick.minor.width": 0.3})
DATA_DIR = "/gpfs/bbp.cscs.ch/home/egassant/reliability_and_structure/data_analysis/data"
CONNECTOMES = ["BBP", "MICrONS", "Celegans", "Drosophila"]
GRAPHS = ["original", "configuration_model", "ER_shuffle", "run_DD2"]
GRAPHS_V2 = ["original", "ER", "distance"]
cmap = plt.cm.Dark2
PALETTE = {"original": "indianred",  # cmap(3),
           "configuration_model": cmap(2),
           "ER": cmap(0),
           "ER_shuffle": cmap(0),
           "distance": cmap(1),
           "run_DD2": cmap(1)}


def load_data():
    """Load data from saved DataFrames and concatenates them (across controls)"""
    conn_data = {}
    for conn in CONNECTOMES:
        dfs = []
        for graph in GRAPHS:
            if graph == "run_DD2" and conn in ["Celegans", "Drosophila"]:
                continue
            df = pd.read_pickle(os.path.join(DATA_DIR, "%s_nbd_basics_%s.pkl" % (conn, graph)))
            df["2_simplices_norm"] = df["2_simplices"] / df["1_simplices"]
            df["graph"] = graph
            dfs.append(df[["graph", "density", "rc_over_nodes", "0_simplices", "2_simplices_norm"]])
        conn_data[conn] = pd.concat(dfs, ignore_index=True)
    return conn_data


def load_data_v2():
    """Load data from saved DataFrames and concatenates them (across controls)"""
    conn_data = {}
    for conn in CONNECTOMES:
        dfs = []
        for graph in GRAPHS_V2:
            if graph == "distance" and conn in ["Celegans", "Drosophila"]:
                continue
            df = pd.read_pickle(os.path.join(DATA_DIR, "props_%s_%s.pkl" % (conn, graph)))
            df["graph"] = graph
            dfs.append(df[["graph", "wasserstein_deg_total", "euclidean_edges_sc"]])
        conn_data[conn] = pd.concat(dfs, ignore_index=True)
    return conn_data


def plot_feature(conn_data, feature, fig_name):
    """Plot feature across connectomes and controls"""
    fig = plt.figure(figsize=(7.5, 1.6))
    for i, conn in enumerate(CONNECTOMES):
        ax = fig.add_subplot(1, len(CONNECTOMES), i + 1)
        sns.histplot(x=feature, hue="graph", bins=21,
                     element="step", palette=PALETTE, legend=False, alpha=0.2, data=conn_data[conn], ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("")
        if fig_name[-3:] == "png":
            ax.set_yscale("log")
    sns.despine(trim=True, offset=1)
    fig.tight_layout()
    fig.savefig(fig_name, bbox_inches="tight", transparent=True, dpi=500)
    plt.close(fig)


if __name__ == "__main__":
    conn_data = load_data()
    for feature in ["0_simplices", "density", "rc_over_nodes", "2_simplices_norm"]:
        ext = "png" if feature in ["0_simplices", "density"] else "pdf"
        plot_feature(conn_data, feature, "figs/%s.%s" % (feature, ext))
    conn_data = load_data_v2()
    for feature in ["wasserstein_deg_total", "euclidean_edges_sc"]:
        plot_feature(conn_data, feature, "figs/%s.pdf" % feature)



