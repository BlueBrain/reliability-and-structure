# SPDX-FileCopyrightText: 2024 Blue Brain Project / EPFL
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from connalysis.network import topology, local
from scipy import stats
import numpy as np
import pandas as pd
from scipy.special import comb
import pickle
from connalysis import randomization 
import os.path



# Metrics to compute on tribes 
def deg_and_skew(adj): 
    degs=topology.node_degree(adj)
    return degs, stats.skew(degs)
def rc_densities(adj):
    rc=topology.rc_submatrix(adj).sum()
    assert adj.shape[0] > 0, "Matrix is empty"
    if adj.sum() == 0: 
        return rc/adj.shape[0], np.nan # reciprocal edges over nodes, rc over edges
    else:
        return rc/adj.shape[0], rc/adj.sum() # reciprocal edges over nodes, rc over edges

from sknetwork.clustering import Louvain, get_modularity
louvain = Louvain()
def modularity(adj):
    if adj.sum()==0: 
        return np.nan
    else:
        labels=louvain.fit_predict(adj)
        return get_modularity(adj,labels)

# Formatting functions of computing metrics of columns of properties computed 
def format_numeric_props(dict): 
    df=pd.concat([pd.DataFrame.from_dict({prop:dict[prop] for prop in ['nbd_size', 'edges'] }),
                  pd.DataFrame.from_dict(dict['rc_densities'], orient="index", columns=["rc_over_nodes", "rc_over_edges"])],
                 axis=1)
    df['density']=df['edges']/(df['nbd_size']*(df['nbd_size']-1))
    return df
def compute_skewness(dict, df, metric='degree'): 
    df[f'{metric}_skewness']=np.nan
    for v in df.index: 
        df[f'{metric}_skewness'].loc[v]=stats.skew(dict[metric][v].sum(axis=1))
    return df
def format_numeric_and_skewness(dict, metric="degree"):
    df=format_numeric_props(dict)
    df=compute_skewness(dict, format_numeric_props(dict))
    return df  

def combine_metrics_to_dict(dict1, dict2, metric, single_col=True, subcols=("IN", "OUT")):
    # Combining metrics of original and control for distance computation 
    if single_col:
        dict_combined={}
        for v in dict1[metric].keys():
            dict_combined[v]=pd.concat([dict1[metric][v], dict2[metric][v]], axis=1).fillna(0).astype(int)
            dict_combined[v].columns=['original', 'control']
        return dict_combined
    else:
        dicts={col:{} for col in subcols}
        for v in dict1[metric].keys():
            df1=dict1[metric][v]; df2=dict2[metric][v]
            for col in subcols:
                dicts[col][v]=pd.concat([df1[col], df2[col]], axis=1).fillna(0).astype(int)
                dicts[col][v].columns=['original', 'control']
        return dicts

def combine_metrics_to_df(dict1, dict2, metric='simplex_counts'):
    # dict1 : dictionary of original network
    # dict2 : dictionary of CM control 
    # metric : scalar metric to combine
    df_ori=pd.DataFrame.from_dict(dict1[metric], orient='index').fillna(0)
    df_control=pd.DataFrame.from_dict(dict2[metric], orient='index').fillna(0)
    # Extend by 0 to have the same number of columns.
    # This makes sense for simplex counts but may not make sense for other metrics!
    m_ori=df_ori.columns.max(); m_control=df_control.columns.max()
    max_dim=max(m_ori,m_control)
    if max_dim>m_control: df_control[np.arange(m_control+1,max_dim +1)]=0
    if max_dim>m_ori: df_ori[np.arange(m_ori+1,max_dim +1)]=0
    return pd.concat([df_ori,df_control], axis=1, keys=['original', 'control'])            
    

### DISTANCE FUNCTIONS ###
# Simplex counts 
def norm_euclid(df, mode):
    # Euclidean distance between normalized simplex counts
    # mode = 'nodes', weights the simplex counts by the number of nodes in a simplex 
    # mode = 'edges', weights the simplex counts by the number of edges in a simplex
    # Normalize simplex counts of control and original jointly be bewteen 0 and 1
    group=df.groupby(axis=1, level=1)
    min_vals=group.min().min(axis=1); max_vals=group.max().max(axis=1)
    normalized_counts=(df-np.expand_dims(min_vals,axis=1)).divide(max_vals-min_vals, axis=0)
    # Weight the vectors to give more relevance to higher dimensional simplices, which reveal more structure
    max_dim=df.columns.get_level_values(level=1).max()
    weights=df.columns.get_level_values(level=1)+1 # Number of nodes in a simplex of that given dimension
    if mode == 'edges': 
        weights=comb(weights,2) # Numbr of edges in a simplex of that dimension
    normalized_counts=normalized_counts*weights
    # Return euclidean distance between normalized vectors
    return np.linalg.norm((normalized_counts.xs("original", level=0, axis=1)-
                           normalized_counts.xs("control", level=0, axis=1)), 
                          axis=1)
def cosine(df):
    # Compute cosine distance between simplex counts
    df_ori=df.xs("original", level=0, axis=1)
    df_control=df.xs("control", level=0, axis=1)
    num=df_ori.multiply(df_control).sum(axis=1)
    den=np.linalg.norm(df_ori, axis=1)*np.linalg.norm(df_control, axis=1)
    return 1-num/den
def compute_distance_metric_df(df_ori, df_metric, metric_name, dist_dict): 
    # Aggregate in DataFrame
    for distance_name in dist_dict.keys(): 
        df_ori[f'{distance_name}_{metric_name}']=dist_dict[distance_name](df_metric)
    return df_ori


# Degree distributions
def wasserstein(df): # This metric is expensive
    x=df.original.values#_counts().sort_index()
    y=df.control.values#_counts().sort_index()
    #return stats.wasserstein_distance(x.index, y.index, x.values, y.values)
    return stats.wasserstein_distance(df.original.values, df.control.values)
def compute_wass_degs(dict_combined, df_ori, direction="total"):
    # Wasserstein distance between degree distributions.  Can this be 
    # made more efficient while keeping the correct wasserstein computation?
    df_ori[f'wasserstein_deg_{direction}']=np.nan
    for v in df_ori.index:
        if direction == 'total': 
            df=dict_combined["IN"][v]+dict_combined["OUT"][v]
        else: 
            df=dict_combined[direction][v]
        df_ori[f'wasserstein_deg_{direction}'].loc[v]=wasserstein(df)
    return df_ori

#### COMPUTE AND SAVE PROPERTIES ACROSS NEIGHBORHOODS ##### 
def compute_properties(analysis_config, base_graph='original', force_recompute=False):
    # Compute properties for all neighborhoods for original and a CM control 
    root=f'{analysis_config["out_file_dir"]}'
    root_raw=f'{root}raw/'
    # Get matrix 
    if base_graph =='original': adj = analysis_config['graph']
    else:
        adj=analysis_config['controls'][base_graph]['function'](**analysis_config['controls'][base_graph]['kwargs'])   
    # Computing properties for the original graph 
    path_out=f'{root_raw}{analysis_config["out_file_prefix"]}_{base_graph}.pkl'
    if (not os.path.isfile(path_out)) or force_recompute:
        computed_props=local.properties_at_neighborhoods(adj, analysis_config['func_config'])
        with open(path_out, 'wb') as f:
            pickle.dump(computed_props, f)
    # Computing properties for the CM control
    path_out=f'{root_raw}{analysis_config["out_file_prefix"]}_{base_graph}_CM.pkl'
    if (not os.path.isfile(path_out)) or force_recompute:
        computed_props_control=local.properties_at_neighborhoods(randomization.configuration_model(adj, 
                                                                                               seed=analysis_config['seed_CM']), 
                                                             analysis_config['func_config'])
        with open(path_out, 'wb') as f:
            pickle.dump(computed_props_control, f)


def format_properties(analysis_config, computed_props, computed_props_control, base_graph='original',save=True):
    # Format properties for analysis 
    root=f'{analysis_config["out_file_dir"]}'
    root_raw=f'{root}raw/'
    # Loading precomputed properties 
    path=f'{root_raw}{analysis_config["out_file_prefix"]}_{base_graph}.pkl'
    with open(path, 'rb') as f:
        computed_props=pickle.load(f)
    path=f'{root_raw}{analysis_config["out_file_prefix"]}_{base_graph}_CM.pkl'
    with open(path, 'rb') as f:
        computed_props_control=pickle.load(f)  
    ### Format data 
    # Collect scalar properties and compute degree skewness 
    df_original=format_numeric_and_skewness(computed_props)
    df_control=format_numeric_and_skewness(computed_props_control)
    # Formatting degree distributions and simplex counts 
    sc=combine_metrics_to_df(computed_props,computed_props_control,metric='simplex_counts')
    degs=combine_metrics_to_dict(computed_props,computed_props_control, metric = 'degree', single_col=False)
    if save: 
        df_original.to_pickle(f'{root}{analysis_config["out_file_prefix"]}_{base_graph}_base.pkl')
        df_control.to_pickle(f'{root}{analysis_config["out_file_prefix"]}_{base_graph}_CM.pkl')
        sc.to_pickle(f'{root}{analysis_config["out_file_prefix"]}_simplex_counts_{base_graph}.pkl')
        with open(f'{root}{analysis_config["out_file_prefix"]}_degrees_{base_graph}.pkl', 'wb') as f:
            pickle.dump(degs, f)
    return df_original, df_control, sc, degs

def compute_all_props(analysis_config, force_recompute=False):
    # Compute properties of the neighborhoods of each network 
    compute_properties(analysis_config, base_graph='original',force_recompute=force_recompute)
    print("Done computing parameters for original graph and CM control")
    for control in analysis_config['controls'].keys():
        compute_properties(analysis_config, base_graph=control,force_recompute=force_recompute)
    print("Done computing parameters for control of original and their CM control")
    
    # Format output and compute distance to CM control
    root=f'{analysis_config["out_file_dir"]}'
    root_raw=f'{root}raw/'
    for base_graph in ['original']+list(analysis_config['controls'].keys()):
        # Load them output of connalysis 
        path=f'{root_raw}{analysis_config["out_file_prefix"]}_{base_graph}.pkl'
        with open(path, 'rb') as f:
            computed_props=pickle.load(f)
        path=f'{root_raw}{analysis_config["out_file_prefix"]}_{base_graph}_CM.pkl'
        with open(path, 'rb') as f:
            computed_props_CM=pickle.load(f)
        df_original, df_CM, sc, degs=format_properties(analysis_config, computed_props, computed_props_CM, base_graph, save=True)
        # Compute simplex counts distances and save
        df_original=compute_distance_metric_df(df_original, sc, 'sc', analysis_config['sc_distances']) 
        df_original.to_pickle(f'{analysis_config["out_file_dir"]}{analysis_config["out_file_prefix"]}_{base_graph}.pkl')
        print("Done computing simplex distances")
        # Compute wasserstein distances 
        for direction in analysis_config['degree_directions']:
            df_original=compute_wass_degs(degs, df_original, direction=direction) 
        df_original.to_pickle(f'{analysis_config["out_file_dir"]}{analysis_config["out_file_prefix"]}_{base_graph}.pkl')
        print("Done computing degree distances")

    
    
