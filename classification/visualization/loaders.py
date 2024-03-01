"""
Functions that load classification accuracies from the PCA and TriDy classification methods and formats them for plotting
authors:Daniela Egas Santander, Jānis Lazovskis
Last update: 02.2024
"""

import numpy as np
import pandas as pd
from scipy import stats
import itertools

#### Getting size of union of selected neighborhoods
def size_of_union_single(centers, df_nbds, return_nodes=False):
    union=np.unique(np.concatenate([centers]+df_nbds.loc[centers].to_list()).astype(int))
    if return_nodes:
        return union.shape[0], union
    else:
        return union.shape[0]
    
def load_centers_and_size_union(df_nbds, fname):
    df_centers=pd.read_pickle(fname)
    union_sizes=[]
    for i in range(len(df_centers)):
        c=df_centers.iloc[i, 3:].to_numpy()
        union_sizes.append(size_of_union_single(c, df_nbds))
    df_centers["size_of_union"]=union_sizes
    return df_centers

####  Labelling PCA data and adding basic stats 
def split_indices_PCA(data): 
    index_double_nodes=np.empty(0)
    index_double_edges=np.empty(0)
    index_single=np.empty(0)
    for key in data.index:
        if key[0:19]=='rc_over_nodes_then_':
            index_double_nodes=np.append(index_double_nodes, key)
            index_single=np.append(index_single, key[19:])
        elif key[0:18]=='rc_per_edges_then_':
            index_double_edges=np.append(index_double_edges, key)
            index_single=np.append(index_single, key[18:])
        index_single=np.unique(index_single)
    return data, index_single, index_double_nodes, index_double_edges

def load_and_stats_PCA(fname):
    data, index_single, index_double, index_double_edges=split_indices_PCA(pd.read_pickle(fname))
    controls=data.drop(np.concatenate([index_single, index_double, index_double_edges])).index
    # Basic stats
    data=pd.concat([data, pd.DataFrame([data.drop("mean", axis=1).apply(stats.sem, axis=1), 
                                        data.drop("mean", axis=1).apply(np.std, axis=1)],
                                       index=["sem", "std"]).T], axis=1)
    data["variance"]=data["std"]**2

    # Label single and double classfication 
    data['double_classification']=np.nan
    data['parameter_name']=np.nan
    for trial in data.index:
        if np.isin(trial, controls):
            data.loc[trial, 'double_classification']='control'
        elif np.isin(trial, index_double):
            data.loc[trial, 'double_classification']=True
            data.loc[trial, 'parameter_name']=trial[19:]
        elif np.isin(trial, index_double_edges):
            data.loc[trial, 'double_classification']="edges"
            data.loc[trial, 'parameter_name']=trial[18:]
        elif np.isin(trial, index_single):
            data.loc[trial, "double_classification"]=False
            data.loc[trial, 'parameter_name']=trial
    data=data.sort_values(by='parameter_name')
    data.loc[controls,"parameter_name"]=[x[:-2] for x in data.loc[controls].index]
    # Labelling sorting order
    data["selection_order"]="None"
    temp=[x[-6:]=="bottom" for x in data.index]
    data["selection_order"][temp]="bottom"
    temp=[x[-3:]=="top" for x in data.index]
    data["selection_order"][temp]="top"
    
    return data 

#### Reformatting PCA results for plotting 
def get_size_acc_structured_PCA(data,df_chiefs_structured, double_classfication, param, selection_order):
    prefix = "rc_over_nodes_then_" if double_classfication else ''
    index=f"{prefix}{param}_{selection_order}"
    acc_val=data.loc[index, "mean"]
    sem=data.loc[index, "sem"]
    std=data.loc[index, "std"]
    variance=data.loc[index, "variance"]
    if double_classfication:
        chiefs=df_chiefs_structured[~df_chiefs_structured["first_selection"].isna()]
    else:
        chiefs=df_chiefs_structured[df_chiefs_structured["first_selection"].isna()]
    size=chiefs.query(f"second_selection=='{param}' and selection_order=='{selection_order}'")["size_of_union"].iloc[0]
    return acc_val, sem, std, variance, size

def get_plot_df_structured_PCA(data, df_chiefs_structured):

    plot_me=pd.DataFrame(index=pd.MultiIndex.from_tuples(
        itertools.product(["bottom", "top"], df_chiefs_structured["second_selection"].unique())),
                         columns=pd.MultiIndex.from_tuples(itertools.product(["single", "double"],
                                                                             ["acc", "sem","std", "variance", "size"])))

    for selection_order in ["bottom", "top"]:
        for param in df_chiefs_structured["second_selection"].unique():
            plot_me.loc[(selection_order,param)]=np.concatenate([get_size_acc_structured_PCA(data,df_chiefs_structured, 
                                                                                   double_classfication=False, 
                                                                                   param=param, 
                                                                                   selection_order=selection_order),
                                                          get_size_acc_structured_PCA(data,df_chiefs_structured, 
                                                                                  double_classfication=True, 
                                                                                  param=param, 
                                                                                  selection_order=selection_order)])
    return plot_me

def get_size_acc_random_PCA(data, control_type, control_no, df_chiefs_random, connectome, nbds):
    index=f"{control_type}_{control_no}"
    acc_val=data.loc[index, "mean"]
    sem=data.loc[index, "sem"]
    std=data.loc[index, "std"]
    variance=data.loc[index, "variance"]
    sel_centers=np.array([connectome.vertices[connectome.vertices["index"]==x].index[0] for x in df_chiefs_random.query(f"{index}==1").index])
    size=size_of_union_single(sel_centers, nbds)
    return acc_val, sem, std, variance, size
def get_plot_df_random_PCA(data, df_chiefs_random, connectome, df_nbds):
    plot_me_random=pd.DataFrame(index=pd.MultiIndex.from_tuples(
        itertools.product(["random", "random_dense", "random_sparse"], range(3))),
                         columns=["acc","sem", "std","variance", "size"])
    
    for control_type in ["random", "random_dense", "random_sparse"]:
        for control_no in range(3):
            plot_me_random.loc[(control_type,control_no)]=get_size_acc_random_PCA(data, control_type, control_no, df_chiefs_random, connectome, df_nbds)
    return plot_me_random   

def reformat_PCA_results(fname_centers_structured, fname_centers_random, fname_acc, df_nbds, connectome): 
    df_chiefs_structured=load_centers_and_size_union(df_nbds, fname_centers_structured)
    df_chiefs_random=pd.read_pickle(fname_centers_random)
    data=load_and_stats_PCA(fname_acc)
    return (get_plot_df_structured_PCA(data, df_chiefs_structured),
            get_plot_df_random_PCA(data, df_chiefs_random, connectome, df_nbds))
    

#### Formatting TriDy data 
def load_acc_tridy(prefix, df_centers):
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
    params = np.unique(df_centers['first_selection'].values)
    params = np.array([p for p in params if p not in ['rc_per_edges','rc_per_nodes']])
    results= {}
    for p in fparams_short2long:
        results[p] = pd.read_pickle(prefix+p+'.pkl').sort_values(by='bin_number').reset_index(drop=True)
    return results
    
def get_plot_df_structured_tridy(df_centers, df_acc):
    data=pd.concat([df_centers.iloc[:,[0,1,2,-1]], df_acc], axis=1)

    # Aggregate single selection data 
    df_single=[]
    df=data[data["second_selection"].isna()].query("first_selection!='rc_per_nodes'")
    for selection_order in ["bottom", "top"]:
        df_single.append(df.query(f"selection_order=='{selection_order}'").set_index(["selection_order", "first_selection"]))
    df_single=pd.concat(df_single).drop("second_selection", axis=1)
    # Aggregate double selectiondata for rc per nodes
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