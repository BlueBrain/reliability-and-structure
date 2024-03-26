"""
Activity plots
author: András Ecker, last update: 02.2024
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from bluepy import Simulation
import sys
sys.path.append("../../../library")
from structural_basic import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="ticks", context="paper", font="Helvetica Neue",
        rc={"axes.labelsize": 7, "legend.fontsize": 6, "axes.linewidth": 0.6, "xtick.labelsize": 6, "ytick.labelsize": 6,
            "xtick.major.size": 2, "xtick.major.width": 0.5, "xtick.minor.size": 1.5, "xtick.minor.width": 0.3,
            "ytick.major.size": 2, "ytick.major.width": 0.5, "ytick.minor.size": 1.5, "ytick.minor.width": 0.3})
RED, BLUE = "#e32b14", "#3271b8"
STRUCTURAL_DATA_DIR = "/gpfs/bbp.cscs.ch/home/egassant/reliability_and_structure/data_analysis/data"
MICRONS_FN_DATA_DIR = "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/assemblyfire/MICrONS"
BBP_FN_DATA_DIR = "/gpfs/bbp.cscs.ch/data/scratch/proj9/bisimplices/bbp_workflow/"
LAYER_MTYPES = {23: ["23P"], 4: ["4P"], 5: ["5P_IT", "5P_NT", "5P_PT"], 6: ["6CT", "6IT"]}
CLIPS = ["Clip/C", "Clip/R", "Clip/S", "Monet2", "Trippy"]
cmap = plt.cm.get_cmap("tab10", len(CLIPS))
CLIP_COLORS = {clip: matplotlib.colors.to_hex(cmap(i)) for i, clip in enumerate(CLIPS)}
PATTERNS = ["A", "B", "C", "D", "E", "F", "G", "H"]
cmap = plt.cm.get_cmap("tab10", len(PATTERNS))
PATTERN_COLORS = {pattern: matplotlib.colors.to_hex(cmap(i)) for i, pattern in enumerate(PATTERNS)}


def setup_imshow(npzf_name, t_start, t_end, conn_mat):
    """Loads deconvolved spikes from MICrONS (saved to npz in `assemblyfire`)
    maps idx to the structural connectome and orders cells based on mtypes"""
    tmp = np.load(npzf_name)
    df = pd.DataFrame(tmp["spikes"], index=tmp["idx"], columns=tmp["t"])
    df = df.loc[:, (t_start <= df.columns) & (df.columns < t_end)]
    stim_times, clips = tmp["stim_times"], tmp["pattern_names"]
    idx = np.where((t_start < stim_times) & (stim_times < t_end))[0]
    stim_times, clips = stim_times[idx], clips[idx]
    # match idx to structural data
    df.loc[df.index.drop_duplicates(keep=False)]  # drop duplicate idx
    valid_idx = conn_mat.vertices.id[np.isin(conn_mat.vertices.id, df.index)]
    df = df.loc[df.index.isin(valid_idx)]
    # sort by mtype
    valid_idx_tmp = pd.Series(valid_idx.index.to_numpy(), index=valid_idx.to_numpy())
    mtypes = conn_mat.vertices["cell_type"].loc[valid_idx_tmp[df.index].to_numpy()].sort_values()
    data = df.loc[valid_idx[mtypes.index.to_numpy()].to_numpy()].to_numpy()
    # get layer boundaries
    mtypes.reset_index(drop=True, inplace=True)
    yticks, yticklabels = [], []
    for layer, layer_mtypes in LAYER_MTYPES.items():
        yticks.append(int(mtypes.loc[mtypes.isin(layer_mtypes)].index.to_numpy().mean()))
        yticklabels.append("L%s" % layer)
    return data, df.columns.to_numpy(), stim_times, clips, yticks, yticklabels


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

    return spike_times - t_start, spiking_ys, cols, rates, [0, t_end - t_start], ylim, yticks, yticklabels


def setup_input_raster(sim, t_start, t_end):
    """Organize gids by rate and get stimulus train"""
    # spikes
    input_fname = os.path.abspath(os.path.join(BBP_FN_DATA_DIR, "7b381e96-91ac-4ddd-887b-1f563872bd1c", "0",
                                               sim.config["Stimulus_spikeReplay"]["SpikeFile"]))
    spikes = pd.read_csv(input_fname, sep="\t").reset_index()
    spikes.rename(columns={"index": "t", "/scatter": "gid"}, inplace=True)
    spikes = spikes.loc[(t_start <= spikes["t"]) & (spikes["t"] <= t_end)]
    spike_times = spikes["t"].to_numpy()
    n_spikes = spikes["gid"].value_counts().to_frame()
    n_spikes["new_gid"] = np.arange(len(n_spikes))
    spikes["new_gid"] = n_spikes.loc[spikes["gid"], "new_gid"].to_numpy()
    rate = _calc_rate(spike_times, len(n_spikes), t_start, t_end)
    # stimulus train and fiber locs
    input_config = os.path.splitext(input_fname)[0] + ".json"
    with open(input_config, "r") as f:
        stim_config = json.load(f)
    stims = pd.DataFrame(np.array(stim_config["props"]["time_windows"][:-1]), columns=["t"])
    stims["id"] = np.array(stim_config["props"]["stim_train"])
    stims = stims.loc[(t_start <= stims["t"]) & (stims["t"] < t_end)]
    stims["name"] = stims["id"].map({i: name for i, name in enumerate(PATTERNS)})

    return spike_times - t_start, spikes["new_gid"].to_numpy(), rate, [0, t_end - t_start], [-1, len(n_spikes)], \
           stims["t"].to_numpy() - t_start, stims["name"].to_numpy(), stim_config["cfg"]["duration_stim"], \
           stim_config["props"]["pattern_grps"], np.array(stim_config["props"]["grp_pos"])


def plot_imshow(data, t, stim_times, clips, yticks, yticklabels, fig_name):
    """Deconvolved spike traces, rate, and clips shown"""
    fig = plt.figure(figsize=(9, 2))
    ax = fig.add_subplot(1, 1, 1)
    i = ax.imshow(data, cmap="Reds", norm=matplotlib.colors.LogNorm(), aspect="auto", interpolation=None)
    plt.colorbar(i)
    ax2 = ax.twinx()
    ax2.plot(gaussian_filter1d(np.mean(data, axis=0), 2), color="black")
    for i, (t_start, t_end) in enumerate(zip(stim_times[:-1], stim_times[1:])):  # the last won't show...
        idx = [np.searchsorted(t, t_start), np.searchsorted(t, t_end)]
        ax.plot(idx, [1, 1], CLIP_COLORS[clips[i]], lw=3)
        ax.text(int(np.mean(idx)), -1, clips[i], ha="center", va="bottom", fontsize=7, weight="bold")
    ax.set_xticks(np.linspace(0, data.shape[1], 8))
    ax.set_xticklabels(np.around((np.linspace(t[0], t[-1], 8) - t[0]) / 5).astype(int) * 5)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax2.set_yticks([])
    fig.savefig(fig_name, bbox_inches="tight", transparent=True, dpi=500)
    plt.close(fig)


def plot_raster(spike_times, spiking_ys, cols, rates, xlim, ylim, yticks, yticklabels, fig_name, fig_size=(8, 2)):
    """Raster and firing rates"""
    t_rate = np.linspace(xlim[0], xlim[1], len(rates["EXC"]))
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor((0.95, 0.95, 0.95))
    ax.scatter(spike_times, spiking_ys, c=cols, alpha=0.9, marker='.', s=0.5, edgecolor="none")
    ax2 = ax.twinx()
    ax2.plot(t_rate, rates["EXC"], RED)
    ax2.plot(t_rate, rates["INH"], BLUE)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_yticks(yticks)
    ax2.set_ylim(bottom=0)
    ax.set_yticklabels(yticklabels)
    fig.savefig(fig_name, bbox_inches="tight", transparent=True, dpi=300)
    plt.close(fig)


def plot_input_raster(spike_times, spiking_ys, rate, xlim, ylim,
                      stim_times, stims, stim_dur, pattern_grps, grp_pos, fig_dir):
    """Raster, firing rate, and stimulus train"""
    # raster
    fig = plt.figure(figsize=(8, 0.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor((0.95, 0.95, 0.95))
    ax.scatter(spike_times, spiking_ys, c="black", marker='.', s=3, edgecolor="none")
    ax2 = ax.twinx()
    ax2.plot(np.linspace(xlim[0], xlim[1], len(rate)), rate, "black")
    for t, stim in zip(stim_times, stims):
        ax.plot([t, t + stim_dur], [ylim[-1] + 5, ylim[-1] + 5], PATTERN_COLORS[stim], lw=3)
        ax.text(t + stim_dur / 2, ylim[-1] + 12, stim, ha="center", va="bottom", fontsize=7, weight="bold")
    ax.set_xlim(xlim)
    ax.set_xticks([])
    ax.set_ylim(bottom=ylim[0])
    ax.set_yticks([])
    ax2.set_ylim(bottom=0)
    sns.despine(right=False)
    fig.savefig(fig_dir + "BBP_input_raster.png", bbox_inches="tight", transparent=True, dpi=500)
    plt.close(fig)
    # input fiber(bundle) locations
    for i, pattern in enumerate(PATTERNS):
        fig = plt.figure(figsize=(0.8, 0.8))
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(grp_pos[:, 0], grp_pos[:, 2], c="darkgrey", marker='.', s=20, edgecolor="none")
        ax.scatter(grp_pos[pattern_grps[i], 0], grp_pos[pattern_grps[i], 2], c=PATTERN_COLORS[pattern],
                   marker='.', s=40, edgecolor="none")
        ax.text(np.min(plt.xlim()), np.max(plt.ylim()), " %s" % pattern, color=PATTERN_COLORS[pattern],
                 ha="left", va="top", fontsize=7, weight="bold")
        ax.axis("equal")
        ax.axis("off")
        fig.savefig(fig_dir + "pattern_%s.png" % pattern, bbox_inches="tight", transparent=True, dpi=500)
        plt.close(fig)


if __name__ == "__main__":
    session_id, scan_id = 4, 7
    t_start, t_end = 110, 210  # s
    npzf_name = os.path.join(MICRONS_FN_DATA_DIR, "MICrONS_session%i_scan%i.npz" % (session_id, scan_id))
    plot_imshow(*setup_imshow(npzf_name, t_start, t_end, load_connectome(STRUCTURAL_DATA_DIR, "MICrONS")),
                "figs/paper/MICrONS_session%i_scan%i.png" % (session_id, scan_id))

    t_start, t_end = 19200, 21200  # ms
    sim = Simulation(os.path.join(BBP_FN_DATA_DIR, "7b381e96-91ac-4ddd-887b-1f563872bd1c", "0", "BlueConfig"))
    plot_raster(*setup_raster(sim, t_start, t_end), "figs/paper/BBP_raster.png")
    plot_input_raster(*setup_input_raster(sim, t_start, t_end), "figs/paper/")

    t_start, t_end = 3000, 5000
    sim = Simulation(os.path.join(BBP_FN_DATA_DIR, "7ea326a9-79c8-4a24-93c3-207c89629fdf", "0", "BlueConfig"))
    plot_raster(*setup_raster(sim, t_start, t_end), "figs/paper/RC+5_raster.png", fig_size=(4., 1.6))
    sim = Simulation(os.path.join(BBP_FN_DATA_DIR, "364338ae-7913-4790-8d3a-3080fea42633", "0", "BlueConfig"))
    plot_raster(*setup_raster(sim, t_start, t_end), "figs/paper/RC+5_ctrl_raster.png", fig_size=(4., 1.6))
    sim = Simulation(os.path.join(BBP_FN_DATA_DIR, "ab6764bb-9f0e-47d7-84ac-6b5114a587e5", "0", "BlueConfig"))
    plot_raster(*setup_raster(sim, t_start, t_end), "figs/paper/RC+6_raster.png", fig_size=(4., 1.6))
    sim = Simulation(os.path.join(BBP_FN_DATA_DIR, "d4e2b48e-2faa-46cf-a099-2489f10a45e8", "0", "BlueConfig"))
    plot_raster(*setup_raster(sim, t_start, t_end), "figs/paper/RC+6_ctrl_raster.png", fig_size=(4., 1.6))


