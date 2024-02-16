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
from coupling_coefficient import normalize_CC_spikes, normalize_CC_traces

def add_layers(conn, connectome):
    # Add layers as property for grouping
    if conn=="MICrONS":
        layer_mytpes={"23P":"L23",
              "4P":"L4",
              "5P_IT":"L5",
              "5P_NP":"L5",
              "5P_PT":"L5",
              "6IT":"L6",
              "6CT":"L6", 
              'BPC':np.nan
             }
        connectome.add_vertex_property(new_label="layer_group",  
                                               new_values=[layer_mytpes[x] for x in connectome.vertices["cell_type"]])
    elif conn=="BBP":
        layer_grouping={2: "L23", 
               3: "L23", 
               4:"L4", 
               5:"L5", 
               6:"L6"}
        connectome.add_vertex_property(new_label="layer_group",  
                                           new_values=[layer_grouping[x] for x in connectome.vertices["layer"]])
    return connectome
def add_reliability(connectome, conn, fname):
    # Add reliability values 
    if conn=="MICrONS":
        df_act=pd.read_pickle(fname)
        connectome.add_vertex_property(new_label="reliability",
                                       new_values= (stats.zscore(df_act.T.xs('oracle_score', level=1), axis=1,
                                                                 nan_policy="omit").mean(axis=0)).to_numpy())
    elif conn=="BBP":
        connectome.add_vertex_property(new_label="reliability",  
                                       new_values=np.load(fname)['reliability'])
    return connectome
    
def add_cc(connectome, conn, fname, norm_type, bin_size=None):
    # Add CC
    prop_names={"global":"CC",  "per_cell":"CC_norm_cell"}
    if conn=="MICrONS":
        vals=normalize_CC_traces(fname, norm_type=norm_type).to_numpy()
    elif conn=="BBP":
        index=connectome.vertices["index"]
        vals=normalize_CC_spikes(fname, bin_size, norm_type=norm_type, reindex=True, index=index).to_numpy()
    connectome.add_vertex_property(new_label=prop_names[norm_type],new_values=vals)
    return connectome