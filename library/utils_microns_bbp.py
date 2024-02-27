''' 
Utitly functions in ordder to add activity based node property conn ConnetivityMatrix objects for BBP and MICrONS
Author(s): Daniela Egas S. 
Last update: 02.2024 
'''

# General 
import numpy as np
import pandas as pd
from scipy import stats
# Connectomes and network analysis
import conntility
# Specific to this project
import sys 
sys.path.append('../../../library')
from structural_basic import load_connectome
from preprocess import load_spike_trains
from coupling_coefficient import normalize_cc


def add_layers(connectome, conn):
    """Add layers as property for grouping"""
    if conn == "MICrONS":
        layer_mytpes = {"23P": "L23", "4P": "L4", "5P_IT": "L5", "5P_NP": "L5", "5P_PT": "L5",
                        "6IT": "L6", "6CT": "L6", "BPC": np.nan}
        connectome.add_vertex_property(new_label="layer_group",
                                       new_values=[layer_mytpes[x] for x in connectome.vertices["cell_type"]])
    elif conn == "BBP":
        layer_grouping={2: "L23", 3: "L23", 4: "L4", 5: "L5", 6: "L6"}
        connectome.add_vertex_property(new_label="layer_group",
                                       new_values=[layer_grouping[x] for x in connectome.vertices["layer"]])
    return connectome


def add_firing_rates(connectome, conn, fname, format=None):
    if conn == "BBP" and format == "toposample":
        spikes, gids, t_max = load_spike_trains(fname)
        gid, nspikes = np.unique(spikes[:, 1], return_counts=True)
        rates = pd.Series(nspikes, index=gid).reindex(connectome.vertices["index"]) / t_max
        connectome.add_vertex_property(new_label="rates", new_values=rates.to_numpy())
    elif conn == "MICrONS":
        df_act = pd.read_pickle(fname)
        connectome.add_vertex_property(new_label="rates", new_values=df_act.T.xs("rate", level=1).mean().to_numpy())
    return connectome


def add_reliability(connectome, conn, fname):
    if conn == "MICrONS":
        df_act = pd.read_pickle(fname)
        connectome.add_vertex_property(new_label="reliability",
                                       new_values=(stats.zscore(df_act.T.xs("oracle_score", level=1), axis=1,
                                                                nan_policy="omit").mean(axis=0)).to_numpy())
    elif conn == "BBP":
        connectome.add_vertex_property(new_label="reliability", new_values=np.load(fname)["reliability"])
    return connectome


def add_cc(connectome, fname, norm_type):
    prop_names = {"global": "CC", "per_cell": "CC_norm_cell"}
    connectome.add_vertex_property(new_label=prop_names[norm_type],
                                   new_values=normalize_cc(fname, norm_type=norm_type).to_numpy())
    return connectome


def add_efficiency(connectome, fname):
    dims_df = pd.read_pickle(fname)
    for col in dims_df.columns:
        connectome.add_vertex_property(new_label=col, new_values=dims_df[col].to_numpy())
    df = connectome.vertices
    connectome.add_vertex_property(new_label="efficiency",
                                   new_values=(df["actitivy_dimension"] / df["active_ts"]).astype("float").to_numpy())
    return connectome