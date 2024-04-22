# SPDX-FileCopyrightText: 2024 Blue Brain Project / EPFL
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Functions that load classification accuracies from the PCA and TriDy classification methods and formats them for plotting
authors:Daniela Egas Santander, Jānis Lazovskis
Last update: 02.2024
"""

import numpy as np
import pandas as pd
from scipy import stats
import itertools
import conntility

#### Getting size of union of selected neighborhoods
def size_of_union_single(centers, df_nbds, return_nodes=False):
    union=np.unique(np.concatenate([centers]+df_nbds.loc[centers].to_list()).astype(int))
    if return_nodes:
        return union.shape[0], union
    else:
        return union.shape[0]
    
def load_centers_and_size_union(df_nbds, df_centers):
    #df_centers=pd.read_pickle(fname)
    union_sizes=[]
    for i in range(len(df_centers)):
        c=df_centers.iloc[i].loc[[f"chief{k}" for k in range(50)]].to_numpy()
        union_sizes.append(size_of_union_single(c, df_nbds))
    df_centers["size_of_union"]=union_sizes
    return df_centers

####  Reformatting PCA data and adding basic stats for plotting
def split_indices_PCA(df):
    """Split indices of the df frame into single classfication, 
    double classficiation on edges per nodes or 
    double classficiation on edges per edges"""
    index_double_nodes=np.empty(0)
    index_double_edges=np.empty(0)
    index_single=np.empty(0)
    for key in df.index:
        if key[0:19]=='rc_over_nodes_then_':
            index_double_nodes=np.append(index_double_nodes, key)
            index_single=np.append(index_single, key[19:])
        elif key[0:18]=='rc_per_edges_then_':
            index_double_edges=np.append(index_double_edges, key)
            index_single=np.append(index_single, key[18:])
        index_single=np.unique(index_single)
    controls=df.drop(np.concatenate([index_single, index_double_nodes, index_double_edges])).index
    return df, index_single, index_double_nodes, index_double_edges, controls

def reformat_PCA_df(df):
    df, index_single, index_double, index_double_edges, controls=split_indices_PCA(df)
    # Labelling sorting order
    df["selection_order"]="None"
    # Label single and double classfication and controls
    df['double_classification']=np.nan
    df['parameter_name']=np.nan
    for trial in df.index:
        if np.isin(trial, controls):
            df.loc[trial, 'double_classification']='control'
        elif np.isin(trial, index_double):
            df.loc[trial, 'double_classification']=True
            if trial[-6:]=="bottom":
                df.loc[trial, 'selection_order']="bottom"
                df.loc[trial, 'parameter_name']=trial[19:-7]
            elif trial[-3:]=="top":
                df.loc[trial, 'selection_order']="top"
                df.loc[trial, 'parameter_name']=trial[19:-4]
        elif np.isin(trial, index_double_edges):
            df.loc[trial, 'double_classification']="edges"
            df.loc[trial, 'parameter_name']=trial[18:]
        elif np.isin(trial, index_single):
            df.loc[trial, "double_classification"]=False
            if trial[-6:]=="bottom":
                df.loc[trial, 'selection_order']="bottom"
                df.loc[trial, 'parameter_name']=trial[:-7]
            elif trial[-3:]=="top":
                df.loc[trial, 'selection_order']="top"
                df.loc[trial, 'parameter_name']=trial[:-4]
    df=df.sort_values(by='parameter_name')
    df.loc[controls,"parameter_name"]=[x[:-2] for x in df.loc[controls].index]
    # Labelling sorting order
    df["selection_order"]="None"
    temp=[x[-6:]=="bottom" for x in df.index]
    df["selection_order"][temp]="bottom"
    temp=[x[-3:]=="top" for x in df.index]
    df["selection_order"][temp]="top"
    
    return df

def load_and_stats_PCA(fname):
    #data = reformat_PCA_df(pd.read_pickle(fname))
    data = pd.read_pickle(fname)
    # Basic stats
    data=pd.concat([data, pd.DataFrame([data.loc[:, np.arange(6)].apply(stats.sem, axis=1), 
                                        data.loc[:, np.arange(6)].apply(np.std, axis=1)],
                                       index=["sem", "std"]).T], axis=1)
    data["variance"]=data["std"]**2

    return data 
def load_selection_and_get_size_PCA(conn_dir, fname, df_nbds):
    selections=pd.read_pickle(fname)
    trials=selections.columns
    selections=selections.reset_index()
    # Full matrix for re-indexing
    conn_full=conntility.ConnectivityMatrix.from_h5(f'{conn_dir}/connectome_BBP_V5.h5')  
    selections=selections[conn_full.vertices["synapse_class"]=="EXC"].reset_index(names="index_in_full")
    selections=pd.DataFrame(np.array([selections.index[selections[col]==1] for col in selections.columns[2:]]), 
                            columns=[f"chief{k}" for k in range(50)], index=trials)
    selections=load_centers_and_size_union(df_nbds, selections)
    selections=reformat_PCA_df(selections)
    return selections

def get_size_acc_PCA(data,selections, double_classfication, param, selection_order):
    #prefix = "rc_over_nodes_then_" if (double_classfication==True) else ''
    if double_classfication=="control":
        index=f"{param}"
    elif  double_classfication:
        index=f"rc_over_nodes_then_{param}_{selection_order}"
    else:
        index=f"{param}_{selection_order}"
    acc_val=data.loc[index, "mean"]
    sem=data.loc[index, "sem"]
    std=data.loc[index, "std"]
    variance=data.loc[index, "variance"]
    size=selections.loc[index,"size_of_union"]
    return acc_val, sem, std, variance, size
    
def get_plot_df_structured_PCA(data, selections):
    # Restrict to structured selection 
    selections=selections.sort_index()
    data=data.sort_index()
    data=data[selections["double_classification"]!="control"]
    selections=selections[selections["double_classification"]!="control"]
    # Reformat 
    plot_me=pd.DataFrame(index=pd.MultiIndex.from_tuples(
        itertools.product(["bottom", "top"], selections["parameter_name"].unique())),
                         columns=pd.MultiIndex.from_tuples(itertools.product(["single", "double"],
                                                                             ["acc", "sem","std", "variance", "size"])))

    for selection_order in ["bottom", "top"]:
        for param in selections["parameter_name"].unique():
            plot_me.loc[(selection_order,param)]=np.concatenate([get_size_acc_PCA(data,selections, 
                                                                                   double_classfication=False, 
                                                                                   param=param, 
                                                                                   selection_order=selection_order),
                                                          get_size_acc_PCA(data,selections, 
                                                                                  double_classfication=True, 
                                                                                  param=param, 
                                                                                  selection_order=selection_order)])
    return plot_me

def get_plot_df_controls_PCA(data, selections):
    df=pd.concat([selections, data], axis=1)
    df=df[df["double_classification"]=="control"].loc[:,["mean", "sem","std", "variance", "size_of_union"]]
    df=df.rename(columns={"mean":"acc", "size_of_union":"size"})
    index=[x[::-1].split("_",1) for x in df.index]
    df.index=pd.MultiIndex.from_tuples([(x[1][::-1], x[0]) for x in index])
    return df

def reformat_PCA_results(full_conn_dir, data_dir, df_nbds):
    """ Load and reformat data for plotting 
    full_conn_dir: directory for the full ConnectivityMatrix
    fname_selection: path to the DataFrame of selections 
    fname_results: path to the DataFrame of accuracy results
    df_nbds: DataFrame of nbd sizes within the E-E subcircuit"""
    fname_selection=f"{data_dir}/PCA/community_database_PCA.pkl"
    fname_acc=f"{data_dir}/PCA/classification_results_PCA.pkl"
    selections = load_selection_and_get_size_PCA(full_conn_dir, fname_selection, df_nbds)
    data=load_and_stats_PCA(fname_acc)
    return (get_plot_df_structured_PCA(data, selections), get_plot_df_controls_PCA(data, selections))
    

#### Formatting TriDy data 
fparams_short2long = {# Featurization paramters
        'asg':'asg', 'asl':'asl', 'asr':'asr',
        'blsg':'blsg', 'blsl':'blsg_low', 'blsr':'blsg_radius',
        'blsgR':'blsg_reversed', 'blsRr':'blsg_reversed_radius', 'blsRl':'blsg_reversed_low',
        'clsg':'clsg', 'clsh':'clsg_high', 'clsr':'clsg_radius',
        'tpsg':'tpsg', 'tpsl':'tpsg_low', 'tpsr':'tpsg_radius',
        'tpsgR':'tpsg_reversed', 'tpsRr':'tpsg_reversed_radius', 'tpsRl':'tpsg_reversed_low',
         'dc2':'dc2', 'dc3':'dc3', 'dc4':'dc4', 'dc5':'dc5', 'dc6':'dc6',
        'nbc':'nbc', 'ec':'ec', 'fcc':'fcc', 'tcc':'tcc',
        'ts':'tribe_size', 'deg':'deg', 'ideg':'in_deg','odeg':'out_deg'#, 
        #'rc':'rc', 'rcc':'rc_chief'
        
    }
def load_acc_tridy(data_dir):
    results_acc= {}
    for p in fparams_short2long.keys():
        fname=f"{data_dir}/network_based/results/reliability-{p}.pkl"
        results_acc[p] = pd.read_pickle(fname).sort_values(by="bin_number").set_index("bin_number")
    return results_acc

def load_selection_and_get_size_TriDy(data_dir, df_nbds):
    df_centers=pd.read_pickle(f"{data_dir}/network_based/selections_reliability.pkl")
    df_centers=load_centers_and_size_union(df_nbds, df_centers)
    df_centers["selection_order"]="Structured"
    # Separate random controls
    df_centers.loc[np.logical_and(df_centers["second_selection"].isna(), 
                                  df_centers["first_selection_order"].isna()), "selection_order"]="random"
    df_centers.loc[np.logical_and(np.logical_and(~df_centers["second_selection"].isna(), df_centers["second_selection_order"].isna()),
                                  df_centers["first_selection_order"]=="bottom"),"selection_order"]="random_sparse"
    df_centers.loc[np.logical_and(np.logical_and(~df_centers["second_selection"].isna(), df_centers["second_selection_order"].isna()),
                                  df_centers["first_selection_order"]=="top"),"selection_order"]="random_dense"
    # Separete single and double selection 
    mask=np.logical_and(df_centers["selection_order"]=="Structured", df_centers["second_selection"].isna())
    df_centers.loc[mask, "selection_order"]=df_centers.loc[mask, "first_selection_order"]
    mask=np.logical_and(df_centers["selection_order"]=="Structured", ~df_centers["second_selection"].isna())
    df_centers.loc[mask, "selection_order"]=df_centers.loc[mask, "second_selection_order"]
    return df_centers

def get_plot_df_structured_tridy(df_centers, df_acc):
    data=pd.concat([df_centers.loc[:,["first_selection","second_selection", "selection_order", "size_of_union"]], df_acc], axis=1)

    # Aggregate single selection data 
    df_single=[]
    df=data[data["second_selection"].isna()].query("first_selection!='rc_per_nodes'")
    for selection_order in ["bottom", "top"]:
        df_single.append(df.query(f"selection_order=='{selection_order}'").set_index(["selection_order", "first_selection"]))
    df_single=pd.concat(df_single).drop("second_selection", axis=1)
    # Aggregate double selection data for rc per nodes
    double_selection=True; df_double=[]
    df=data[~data["second_selection"].isna()]
    for selection_order in ["bottom", "top"]:
        df_double.append(df.query(f"selection_order=='{selection_order}' and first_selection=='rc_per_nodes'").set_index(["selection_order", "second_selection"]))   
    df_double=pd.concat(df_double).drop("first_selection", axis=1)
    # Reformat for plotting
    def relabel_df(df, double_classification):
        df.index.rename(["selection_order", "param"], inplace=True)
        df=df.T.reset_index()
        if double_classification:
            df["double_sel"]="double"
        else:
            df["double_sel"]="single"
        df.set_index(["double_sel", "index"], inplace=True)
        df=df.rename_axis((None, None), axis=0).T
        return df

    return pd.concat([relabel_df(df_single, double_classification=False),
                      relabel_df(df_double, double_classification=True)], axis=1)
def get_plot_df_controls_tridy(df_centers, df_acc):
    # Retrict to random controls
    centers_random=df_centers.query("selection_order=='random' or selection_order=='random_dense' or selection_order=='random_sparse'")
    acc_random=df_acc.loc[centers_random.index]
    df=pd.concat([centers_random.loc[:,["first_selection","second_selection", "selection_order", "size_of_union"]], acc_random], 
                 axis=1)
    df=df.loc[:,['selection_order','size_of_union', 'cv_acc', 'cv_err', 'test_acc', 'test_err','nonzero_count', 'total_count']]
    df=df.rename(columns={"selection_order":"control_type"})
    df["control_number"]="None"
    for ctr_type in df["control_type"].unique():
        df.loc[df["control_type"]==ctr_type, ["control_number"]]=np.arange(len(df[df["control_type"]==ctr_type]))
    df=df.set_index(["control_type", "control_number"])
    return df

def reformat_TriDy_results(data_dir, df_nbds):
    df_centers=load_selection_and_get_size_TriDy(data_dir, df_nbds)
    data=load_acc_tridy(data_dir) 
    results={}
    for p in fparams_short2long.keys(): 
        results[p]={"structured":get_plot_df_structured_tridy(df_centers, data[p]),
                    "controls":get_plot_df_controls_tridy(df_centers, data[p])
                   }
    return results 