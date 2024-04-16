'''
Compute average complexity across (maximal simplices) for all connectomes and their controls.
Runs quickly, can be run interactively.
Author(s): Daniela Egas S.
Last updated: 01.2024
'''

import pandas as pd 
import pickle
from connalysis.network import stats as nstats
    
connectomes=["Celegans", "Drosophila", "MICrONS", "BBP"]
props=["euclidean_edges_sc", "wasserstein_deg_total"]

# Compute average neighborhood complexity across simplices from precomputed data.  This runs quickly, can also be run on the fly.
stats_ori_and_controls={key:None for key in ['original', 'CM', 'ER']}
key_to_name={'original':'original', 'CM':'configuration_model', 'ER':'ER_model'}
seed=10
for base_graph_key in stats_ori_and_controls.keys():
    stats_all={"all":{}, "maximal":{}}
    for conn in connectomes:
        print(f"Analyzing {conn}")
        df_nbd=pd.read_pickle(f"../../data/props_{conn}_{base_graph_key}.pkl")
        for stype, prefix in [("all", ""), ("maximal", "maximal")]:
            stats_all[stype][conn]={}
            s_lists_dict=pd.read_pickle(f"../../data/{conn}_list_simplices_by_dimension_{prefix}.pkl")
            if base_graph_key == "original":
                s_lists = s_lists_dict[base_graph_key]
            else:
                s_lists = s_lists_dict[key_to_name[base_graph_key]][seed]    
            for prop in props:
                vals=df_nbd[prop]
                stats_all[stype][conn][prop]=nstats.node_stats_per_position(s_lists,vals,
                                                                           dims=s_lists.index.drop(0),with_multiplicity=True)
    stats_ori_and_controls[base_graph_key]=stats_all
    print(f"Done with {base_graph_key}")

# Save results
path_out = f"../../data/complexity_by_dimension_all_connectomes.pkl"

with open(path_out, 'wb') as f:
    pickle.dump(stats_ori_and_controls, f)