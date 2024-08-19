import pickle 
import pandas as pd 
import numpy as np

from connalysis.network.classic import closeness_connected_components
from connalysis.network import topology


import sys 
sys.path.append('../../../library')
from structural_basic import load_connectome

# Load pre-computed simplex counts 
root="/gpfs/bbp.cscs.ch/home/egassant/reliability_and_structure/data_analysis/data"
connectomes=["Celegans", "Drosophila", "MICrONS","BBP"]

# Compute closeness and degree distributions.  
# Computing closeness is expensive for the BBP network.  Only need to run once
closeness_precomputed = False
if closeness_precomputed:
    with open(f'{root}/other_node_metrics.pkl', 'rb') as f: 
        node_metrics=pickle.load(f)
else:
    closeness_directed={}; closeness_undirected={}; degs={}
    for conn in connectomes:
        connectome = load_connectome(root, conn)
        adj=connectome.matrix.astype(bool).tocsr()
        adj.setdiag(0)
        adj.eliminate_zeros()
        closeness_directed[conn]=closeness_connected_components(connectome.matrix.astype(bool).tocsr(), directed=True)
        closeness_undirected[conn]=closeness_connected_components(connectome.matrix.astype(bool).tocsr(), directed=False)
        degs[conn] = topology.node_degree(connectome.matrix.astype(bool), direction = ("IN", "OUT"))
        print(f"Done with {conn}")
    # Reformat data and compute node participation from simplex lists 
    node_metrics={}
    for conn in connectomes:
        sl=pd.read_pickle(f"{root}/{conn}_list_simplices_by_dimension_maximal.pkl")["original"]
        node_metrics[conn]=degs[conn].copy()
        node_metrics[conn]["TOTAL"]=node_metrics[conn][["IN", "OUT"]].sum(axis=1)
        node_metrics[conn]["closeness_directed"]=closeness_directed[conn]
        node_metrics[conn]["closeness_undirected"]=closeness_undirected[conn]
        for dim in sl.index.drop(0):
            node, par = np.unique(sl[dim], return_counts=True)
            node_metrics[conn]=pd.concat([node_metrics[conn], pd.DataFrame(par, index=node, columns=[f"par_dim_{dim}"])], axis=1)
        node_metrics[conn].fillna(0,inplace=True)
    with open(f'{root}/other_node_metrics.pkl', 'wb') as f: 
        pickle.dump(node_metrics, f)   