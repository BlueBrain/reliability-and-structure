''' 
Average node properties at nodes using the simplicial structure
Author(s): Daniela Egas S. 
Last update: 02.2024 
'''

# General 
import numpy as np
import pandas as pd
import pickle
from scipy import stats
# Connectomes and network analysis
import conntility
from connalysis.network import stats as nstats
# Specific to this project
import sys 
sys.path.append('../../../library')
from preprocess import load_spike_trains, extract_binned_spike_signals
from dimensionality import get_spectrum_nbds, get_dimensions_nbds, get_dimensions
from structural_basic import *


# Compute poperties across simplices (MOVE TO PRECOMPUTATION?)

def compute_node_stats_per_group(conn,values, group_names, group_values, stypes=["all", "maximal"], base_graph="original"): 
    stats_vals={group:{stype:{conn:{}} for stype in stypes} for group in group_names}
    for stype in stypes: 
        # Load precomputed structural properties 
        if stype =="all":
            s_lists=pd.read_pickle(f"../../data/{conn}_list_simplices_by_dimension_.pkl")[base_graph]
        elif stype =="maximal":
            s_lists=pd.read_pickle(f"../../data/{conn}_list_simplices_by_dimension_maximal.pkl")[base_graph] 
        # Average dimensionality
        for group in group_names:
            # Filter by group
            vals=values.copy()
            vals[group_values!=group]=np.nan
            stats_vals[group][stype]=nstats.node_stats_per_position(s_lists,vals,dims=s_lists.index.drop(0),with_multiplicity=True)
    return stats_vals

# Loading layer data
conns=["MICrONS", "BBP"]
connectomes={}
for conn in conns:
    connectomes[conn]=load_connectome("../../data", conn)

# Adding layer groups to vertex properties
layers=["L23", "L4", "L5", "L6"] 
layer_mytpes={"23P":"L23",
              "4P":"L4",
              "5P_IT":"L5",
              "5P_NP":"L5",
              "5P_PT":"L5",
              "6IT":"L6",
              "6CT":"L6", 
              'BPC':np.nan
             }
layer_grouping={2: "L23", 
               3: "L23", 
               4:"L4", 
               5:"L5", 
               6:"L6"}
connectomes["MICrONS"].add_vertex_property(new_label="layer_group",  
                                           new_values=[layer_mytpes[x] for x in connectomes["MICrONS"].vertices["cell_type"]])
connectomes["BBP"].add_vertex_property(new_label="layer_group",  
                                       new_values=[layer_grouping[x] for x in connectomes["BBP"].vertices["layer"]])


# Loading acivity
# BBP Gaussian kernel values
fname="/gpfs/bbp.cscs.ch/data/scratch/proj9/bisimplices/simulations/BlobStimReliability_O1v5-SONATA_Baseline"
rel_path=f'{fname}/working_dir/reliability_basic.npz'
#zcored values.  Just for testing, these indeed give the same results but shifted
#connectomes["BBP"].add_vertex_property(new_label="reliability",  new_values=stats.zscore(np.load(rel_path)['reliability'], nan_policy="omit"))

# Raw values 
connectomes["BBP"].add_vertex_property(new_label="reliability",  
                                       new_values=np.load(rel_path)['reliability'])
# MICrONS oracle scores
df_act=pd.read_pickle('/gpfs/bbp.cscs.ch/project/proj96/home/ecker/assemblyfire/MICrONS/MICrONS_functional_summary.pkl')
# z-score each session to allow averaging and average
connectomes["MICrONS"].add_vertex_property(new_label="reliability",
                                           new_values= (stats.zscore(df_act.T.xs('oracle_score', level=1), axis=1,
                                                                     nan_policy="omit").mean(axis=0)).to_numpy())

#### Loading Coupling coefficients
# MICrONS
df_act=pd.read_pickle('/gpfs/bbp.cscs.ch/project/proj96/home/ecker/assemblyfire/MICrONS/MICrONS_functional_summary.pkl')
# z-score each session to allow averaging
connectomes["MICrONS"].add_vertex_property(new_label="CC",
                                           new_values=stats.zscore(df_act.T.xs('coupling_coeff', level=1), 
                                                                   axis=1, nan_policy="omit").mean(axis=0).to_numpy())
# BBP 
bin_size="2p0" 
df=pd.read_hdf(
"/gpfs/bbp.cscs.ch/data/scratch/proj9/bisimplices/bbp_workflow/7b381e96-91ac-4ddd-887b-1f563872bd1c/working_dir/coupling_coefficients.h5",
key=f"CC_bin_sz_{bin_size}")
ctr=df.drop("data", axis=1).to_numpy()
# Add z-scored data
vals=((df["data"]-np.mean(ctr))/np.std(ctr)).reindex(connectomes["BBP"].vertices["index"]).to_numpy()
connectomes["BBP"].add_vertex_property(new_label="CC",new_values=vals)

for conn in conns:
    #for property in ["reliability", "CC"]:
    for property in ["CC"]:
        values=connectomes[conn].vertices[property]
        group_names=layers
        group_values=connectomes[conn].vertices["layer_group"].to_numpy()
        output=compute_node_stats_per_group(conn,values, group_names, group_values, stypes=["all", "maximal"], base_graph="original")
        path_out=f"../../data/node_stats_per_layer_{property}_{conn}_original.pkl"
        # Save to pickle
        with open(path_out, 'wb') as fp:
            pickle.dump(output, fp)
            print(f"Done with {conn}, {property}")

