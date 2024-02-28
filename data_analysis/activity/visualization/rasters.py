"""
...
author: András Ecker, last update: 02.2024
"""

import os
import numpy as np
import pandas as pd
from bluepy import Simulation
import sys
sys.path.append("../../../library")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="ticks", context="paper", font="Helvetica Neue",
        rc={"axes.labelsize": 7, "legend.fontsize": 6, "axes.linewidth": 0.6, "xtick.labelsize": 6, "ytick.labelsize": 6,
            "xtick.major.size": 2, "xtick.major.width": 0.5, "xtick.minor.size": 1.5, "xtick.minor.width": 0.3,
            "ytick.major.size": 2, "ytick.major.width": 0.5, "ytick.minor.size": 1.5, "ytick.minor.width": 0.3})
RED, BLUE = "#e32b14", "#3271b8"
MICRONS_FN_DATA_DIR = "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/assemblyfire/MICrONS"
BBP_FN_DATA_DIR = "/gpfs/bbp.cscs.ch/data/scratch/proj9/bisimplices/bbp_workflow/7b381e96-91ac-4ddd-887b-1f563872bd1c"


def _calc_rate(spike_times, n, t_start, t_end, bin_size=10):
    """Calculates populational firing rate"""
    t_bins = np.arange(t_start, t_end+bin_size, bin_size)
    rate, _ = np.histogram(spike_times, t_bins)
    return rate / (n * 1e-3 * bin_size)  # *1e-3 ms to s conversion


def setup_raster(sim, t_start, t_end):
    """Organize gids by depth and color spikes as red/blue for EXC/INH cell (and do this fast)"""
    # structural
    c = sim.circuit
    df = c.cells.get(c.cells.ids(), ["synapse_class", "layer", "y"])
    df["color"] = RED
    df.loc[df["synapse_class"] == "INH", "color"] = BLUE
    ylim = [df["y"].min(), df["y"].max()]  # v5 is inverted...
    yticks, yticklabels = [], []
    for layer in df["layer"].unique():
        yticks.append(df.loc[df["layer"] == layer, "y"].mean())
        yticklabels.append("L%s" % layer)
    # functional
    spikes = sim.spikes.get(t_start=t_start, t_end=t_end)
    spike_times, spiking_gids = spikes.index.to_numpy(), spikes.to_numpy()
    unique_gids, idx = np.unique(spiking_gids, return_inverse=True)
    unique_ys = np.zeros_like(unique_gids, dtype=np.int64)
    unique_cols = np.empty(unique_gids.shape, dtype=object)
    ns = {"EXC": 0, "INH": 0}
    for i, gid in enumerate(unique_gids):
        unique_ys[i] = df.at[gid, "y"]
        unique_cols[i] = df.at[gid, "color"]
        ns[df.at[gid, "synapse_class"]] += 1
    spiking_ys = unique_ys[idx]
    cols = unique_cols[idx]
    rates = {syn_class: _calc_rate(spike_times[cols == color], ns[syn_class], t_start, t_end)
             for syn_class, color in zip(["EXC", "INH"], [RED, BLUE])}

    return spike_times, spiking_ys, cols, rates, ylim, yticks, yticklabels


def plot_raster(spike_times, spiking_ys, cols, rates, ylim, yticks, yticklabels, xlim, fig_name):
    """..."""
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor((0.95, 0.95, 0.95))
    ax.scatter(spike_times, spiking_ys, c=cols, alpha=0.9, marker='.', s=1, edgecolor="none")

    t_rate = np.linspace(t_start, t_end, len(rates["EXC"]))

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    fig.savefig(fig_name, bbox_inches="tight", transparent=True, dpi=500)
    plt.close(fig)


if __name__ == "__main__":
    t_start, t_end = 19200, 21200
    sim = Simulation(os.path.join(BBP_FN_DATA_DIR, "0", "BlueConfig"))
    plot_raster(*setup_raster(sim, t_start, t_end), [t_start, t_end], "figs/paper/BBP_raster.png")









