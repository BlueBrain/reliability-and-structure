# SPDX-FileCopyrightText: 2024 Blue Brain Project / EPFL
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import matplotlib 
import numpy as np
import seaborn as sns
from scipy import stats
import pandas as pd

# Figure settings and color schemes 
titlesize=8
labelsize=6
ticksize=6
conversion=2.54

marker_size=2
linewidth=2; linewidth_base=1; linestyle_base="dotted"; alpha_base=0.75

# Setting up colors 
colors={
    'high complexity': 'tab:purple', 
    'low complexity': 'tab:cyan', 
    2:matplotlib.colormaps["Set3"](0),
    3:matplotlib.colormaps["Set3"](2),
    4:matplotlib.colormaps["Set3"](5),
    5:matplotlib.colormaps["Set3"](4),
    6:matplotlib.colormaps["Set3"](3),
    "all":"C0",
    "sink": "C1",
    "source":"C2"
}
markers={"all":"o",
       "sink": ">",
       "source":"<"}
alpha=0.25

# Plotting functions 
def scatter_and_regress(ax,x, y, color, marker='o', marker_size=marker_size, label=None, color_regress=None, alpha=0.5):
    x=x.to_numpy(); y=y.to_numpy() 
    mask=np.logical_and(~np.isnan(y), ~np.isnan(x))
    regress=stats.linregress(x[mask],y[mask])
    if color_regress is None: color_regress=color
    ax.plot(x, x*regress.slope+regress.intercept, color=color_regress, label=f"{regress.rvalue:.2f}")
    ax.scatter(x, y, c=color, marker=marker,s=marker_size, label=f"{label}",alpha=alpha, zorder=10, rasterized=True )
    h, l = ax.get_legend_handles_labels()
    return ax, h, l

def agg_along_dims_single(stats_dict): 
    # Pull values on source, sink and all positions
    df={conn:{} for conn in stats_dict.keys()}
    for conn in df.keys():
        df[conn]={prop:{} for prop in stats_dict[conn].keys()}
        for prop in stats_dict[conn].keys():
            for stype in stats_dict[conn][prop].keys(): 
                temp={"all":{}, "sink":{}, "source":{}}
                dims=stats_dict[conn][prop][stype].keys()
                dict = stats_dict[conn][prop][stype]
                for dim in dims: 
                    mean=dict[dim]["mean"]
                    err=dict[dim]["sem"]
                    temp['all'][dim]=[mean.loc["all"], err.loc["all"]]
                    temp['source'][dim]=[mean.iloc[0], err.iloc[0]]
                    temp['sink'][dim]=[mean.iloc[-2], err.iloc[-2]]
                df[conn][prop][stype]={key:pd.DataFrame.from_dict(temp[key], orient="index", columns=["mean", "sem"]) 
                                    for key in temp.keys()}
    return df
def agg_along_dims_layers(stats_dict, layers=["L23", "L4", "L5", "L6"]): 
    df={layer:{} for layer in layers}
    for layer in layers: 
        temp={conn:{prop:stats_dict[conn][prop][layer] 
                    for prop in stats_dict[conn].keys() }
              for conn in stats_dict.keys()}
        df[layer]=agg_along_dims_single(temp)
    return df


def plot_violin(ax, y, data,hue="nbd_complexity"):
    sns.violinplot(data=data, x="dummy", y=y, ax=ax,
                   hue=hue, split=True, inner="quart", palette=colors, 
                  hue_order=["low complexity", "high complexity"])#,linewidth=0.001)
    #ax.spines[["left", "top", "bottom"]].set_visible(False)
    #ax.yaxis.set_ticks_position("right")
    #ax.set_yticks([0.6, 0.9],labels=[0.6, 0.9], fontsize=ticksize)
    ax.set_xticks([])
    ax.set_xlabel('')
    #ax.set_ylabel('')'''
    return ax
def plot_and_fill(ax, data, label, color, ms, marker, alpha): 
    ax.plot(data["mean"], marker=marker, label=label, ms=ms)
    ax.fill_between(data.index, data["mean"]-data["sem"], data["mean"]+data["sem"], alpha=alpha)

def kde_and_regress(ax, df, x, y, color):
    x=df[x]; y=df[y] 
    mask=np.logical_and(~np.isnan(y), ~np.isnan(x))
    regress=stats.linregress(x[mask],y[mask])
    ax.plot(x, x*regress.slope+regress.intercept, color=color, label=f"{regress.rvalue:.2f}")
    sns.kdeplot(data=df, x=x, y=y, fill=True, ax=ax, color=color)
    ax.set_ylabel('');     ax.set_xlabel('')
    ax.spines[["right", "top"]].set_visible(False)
    h, l = ax.get_legend_handles_labels()
    return ax, h, l
