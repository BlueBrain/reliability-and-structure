''' 
Average node properties at nodes using the simplicial structure
Author(s): Daniela Egas S. 
Last update: 02.2024 
'''

# General 
import numpy as np
import pandas as pd
import pickle
# Connectomes and network analysis
import conntility
from connalysis.network import stats as nstats
# Specific to this project
import sys 
sys.path.append('../../../library')
from structural_basic import load_connectome
from utils_microns_bbp import *


def compute_node_stats_full(conn, connectome, property, stypes=["all", "maximal"], base_graph="original"): 
    stats_vals={stype:{conn:{}} for stype in stypes}
    for stype in stypes: 
        # Load precomputed simplex lists properties 
        if stype =="all":
            s_lists=pd.read_pickle(f"../../data/{conn}_list_simplices_by_dimension_.pkl")[base_graph]
        elif stype =="maximal":
            s_lists=pd.read_pickle(f"../../data/{conn}_list_simplices_by_dimension_maximal.pkl")[base_graph] 
        # Average each group across simplices
        vals=connectome.vertices[property]
        stats_vals[stype]=nstats.node_stats_per_position(s_lists,vals,dims=s_lists.index.drop(0),with_multiplicity=True)
    return stats_vals
    
def compute_node_stats_per_group(conn, connectome, property, group_name, stypes=["all", "maximal"], base_graph="original"): 
    group_values=connectome.vertices[group_name]
    group_classes=group_values.unique()
    stats_vals={group:{stype:{conn:{}} for stype in stypes} for group in group_classes}
    for stype in stypes: 
        # Load precomputed simplex lists properties 
        if stype =="all":
            s_lists=pd.read_pickle(f"../../data/{conn}_list_simplices_by_dimension_.pkl")[base_graph]
        elif stype =="maximal":
            s_lists=pd.read_pickle(f"../../data/{conn}_list_simplices_by_dimension_maximal.pkl")[base_graph] 
        # Average each group across simplices
        for group in group_classes:
            # Filter by group
            vals=connectome.vertices[property]
            vals.loc[group_values!=group]=np.nan
            stats_vals[group][stype]=nstats.node_stats_per_position(s_lists,vals,dims=s_lists.index.drop(0),with_multiplicity=True)
    return stats_vals
    

configs={}

# Paths to data and specs 
configs["BBP"]={
    "connectome_dir":"../../data",
    "fname_reliability":"/gpfs/bbp.cscs.ch/data/scratch/proj9/bisimplices/simulations/BlobStimReliability_O1v5-SONATA_Baseline/working_dir/reliability_basic.npz",
    "fname_CC": "/gpfs/bbp.cscs.ch/data/scratch/proj9/bisimplices/bbp_workflow/7b381e96-91ac-4ddd-887b-1f563872bd1c/working_dir/coupling_coefficients.h5",
    "norm_types": ["global", "per_cell"],
    "fname_rates":"/gpfs/bbp.cscs.ch/data/scratch/proj9/bisimplices/bbp_workflow/7b381e96-91ac-4ddd-887b-1f563872bd1c/toposample_input/raw_spikes_exc.npy",
    "format_rates":"toposample",
    "bin_size":"2p0", 
    "fname_effciency":"/gpfs/bbp.cscs.ch/data/scratch/proj9/bisimplices/simulations/Toposample_O1v5-SONATA/working_dir/dimensions_EE_from_signals.pkl",
    "save_dir":"../../data", 
    "average_type":["per_layer", "full"], 
    "properties":["CC", "CC_norm_cell", "reliability", "efficiency"], 
    "stypes":["all", "maximal"],
    "base_graph": "original"
}

configs["MICrONS"]={
    "connectome_dir":"../../data",
    "fname_reliability":"/gpfs/bbp.cscs.ch/project/proj96/home/ecker/assemblyfire/MICrONS/MICrONS_functional_summary.pkl",
    "fname_CC": "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/assemblyfire/MICrONS/MICrONS_functional_summary.pkl",
    "norm_types": ["global", "per_cell"],
    "bin_size":None, 
    "save_dir":"../../data", 
    "average_type":["per_layer", "full"], 
    "properties":["CC", "CC_norm_cell", "reliability"], 
    "stypes":["all", "maximal"],
    "base_graph": "original"
}


if __name__ == "__main__":
    for conn in ["MICrONS", "BBP"]:
        cfg=configs[conn]
        stypes=cfg["stypes"]
        base_graph=cfg["base_graph"]
        # Get connectome and porperties per node 
        connectome=load_connectome(cfg["connectome_dir"], conn)
        connectome = add_layers(conn, connectome)
        connectome = add_reliability(connectome, conn, cfg["fname_reliability"])
        for norm_type in cfg["norm_types"]:
            connectome = add_cc(connectome, conn, cfg["fname_CC"], norm_type, bin_size=cfg["bin_size"])
        if conn=="BBP":
            connectome=add_efficiency(conn, connectome, cfg["fname_effciency"])
        print(f"{conn} data loaded")
        # Average selected properties across simplices
        for property in cfg["properties"]:
            if "per_layer" in cfg["average_type"]: # Do per layer analysis 
                group_name="layer_group"
                path_out=f"{cfg['save_dir']}/node_stats_per_layer_{property}_{conn}_{base_graph}.pkl"
                output=compute_node_stats_per_group(conn, connectome, property, group_name, stypes=stypes, base_graph=base_graph)
                # Save to pickle
                with open(path_out, 'wb') as fp:
                    pickle.dump(output, fp)
                    print(f"Done with {conn}, {property} per layer")
            if "full" in cfg["average_type"]: # Do full connectome analysis 
                path_out=f"{cfg['save_dir']}/node_stats_full_{property}_{conn}_{base_graph}.pkl"
                output=compute_node_stats_full(conn, connectome, property, stypes=stypes, base_graph=base_graph)
                # Save to pickle
                with open(path_out, 'wb') as fp:
                    pickle.dump(output, fp)
                    print(f"Done with {conn}, {property} full connectome")
