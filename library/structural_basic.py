'''
Functions to compute basic network properties for a full connectome or on its neighborhoods 
Author(s): Daniela Egas S. 
Last updated: 12.2023
'''

import numpy as np
import pandas as pd
import pickle

from connalysis.network import topology, local 
from connalysis import randomization 
import conntility
from read_connectomes import * # Code obtained from https://github.com/BlueBrain/ConnectomeUtilities/blob/main/examples

def load_connectome(data_dir, connectome): 
    "Loaders and restriction for each of the connectomes"
    if connectome == "Celegans":
        conn=load_C_elegans_stages(data_dir).filter("type").eq("chemical").default(8)
        conn.add_vertex_property('valid_cell', (topology.node_degree(conn.matrix)!=0)) # nodes that are not isolated
        return conn.index("valid_cell").isin(True)     
    elif connectome =="Drosophila":
        return load_drosophila(data_dir).filter("type").eq("axo-dendritic")
    elif connectome == "MICrONS":
        exc_types=['23P', '4P', '5P_IT', '5P_NP', '5P_PT', '6CT', '6IT', 'BPC']
        return load_microns(data_dir).index("cell_type").isin(exc_types)
    elif connectome == "BBP":
        return conntility.ConnectivityMatrix.from_h5(f'{data_dir}/connectome_BBP_V5.h5').index("synapse_class").isin("EXC")

def compute_basic_props(cfg):
    "Compute network properties for connectome and controls" 
    conn=load_connectome(cfg['connectome']['data_dir'], cfg['connectome']['name'])
    M=conn.matrix.astype('bool').tocsr() # remove potential weights of synapses
    print(f"The connectome has {M.diagonal().sum()} entries out of {M.sum()} in the diagonal; these will be ignored.")
    M.setdiag(0)
    M.eliminate_zeros()
    
    for an_type in cfg["analyses"]:
        print(f"Starting {an_type}")
        func=getattr(topology, an_type)
        kwargs=cfg["analyses"][an_type]["kwargs"] if 'kwargs' in cfg["analyses"][an_type] else None
        # Compute analysis on original graph 
        out={}
        out['original']=func(M, **kwargs)
        # Compute analysis on control
        if "controls" in cfg["analyses"][an_type]:
            seeds=cfg["analyses"][an_type]["controls"]["seeds"]
            for ctr_type in cfg["analyses"][an_type]["controls"]["types"]:
                out[ctr_type]={}
                for seed in seeds:
                    f_ctr=getattr(randomization, ctr_type)
                    if ctr_type!='run_DD2':
                        adj=f_ctr(M, seed=seed)
                    else:
                        f_kw=cfg["analyses"][an_type]["controls"]["types"][ctr_type]['kwargs']
                        xyz=conn.vertices[f_kw['xyz_labels']].to_numpy()
                        n=len(conn.vertices)
                        adj=f_ctr(n,f_kw['a'],f_kw['b'],xyz=xyz, seed=(seed, seed))
                    out[ctr_type][seed]=func(adj, kwargs=kwargs)
        path_out=f'{cfg["save_path"]}/{cfg["connectome"]["name"]}_{an_type}_{cfg["analyses"][an_type]["save_suffix"]}.pkl'
        with open(path_out, 'wb') as f: 
            pickle.dump(out,f)
            print(f"Done with {an_type}")

def get_rc_density(adj):
    return topology.rc_submatrix(adj).sum()*100/adj.sum()
    
def rc_original_and_controls(cfg):
    "Compute rc density for original and controls" 
    conn=load_connectome(cfg['connectome']['data_dir'], cfg['connectome']['name'])
    M=conn.matrix.astype('bool').tocsr() # remove potential weights of synapses
    print(f"The connectome has {M.diagonal().sum()} entries out of {M.sum()} in the diagonal; these will be ignored.")
    M.setdiag(0)
    M.eliminate_zeros()

    rc_den={}
    rc_den['original']=get_rc_density(M)
    n_samples=cfg["controls"]["n_samples"]
    for ctr_type in cfg["controls"]["types"]:
        rc_den[ctr_type]=[]
        f_ctr=getattr(randomization, ctr_type)
        if ctr_type!='run_DD2':
            for _ in range(n_samples):
                rc_den[ctr_type].append(get_rc_density(f_ctr(M)))
        else:
            f_kw=cfg["controls"]["types"][ctr_type]['kwargs']
            xyz=conn.vertices[f_kw['xyz_labels']].to_numpy()
            n=len(conn.vertices)
            for _ in range(n_samples):
                rc_den[ctr_type].append(get_rc_density(f_ctr(n,f_kw['a'],f_kw['b'],xyz=xyz)))
        print(f'Done with {ctr_type}')
    path_out=f'{cfg["save_path"]}/{cfg["connectome"]["name"]}_rc_densities.pkl'
    with open(path_out, 'wb') as f: 
        pickle.dump(rc_den,f)
        print(f'Done with {cfg["connectome"]["name"]}')

def compute_basics_over_neighborhoods(M):
    '''Computes, nodes, edges, rc_edges and simplices up to dimension 3 and aggregates in dataframe'''
    func_config={
        'rc_edges':{'function': lambda x : topology.rc_submatrix(x).sum(),
                    'kwargs': {}},
        'edges':{'function': lambda x : x.sum(),
    
                                           'kwargs': {}},
        'nodes':{'function':lambda x : x.shape[0],
    
                           'kwargs': {}}, 
        'simplex_counts':{'function':topology.simplex_counts,
    
                                     'kwargs': {"threads": 8, "max_dim":3}
                                    }
    }
    out=local.properties_at_neighborhoods(M, func_config)
    props=pd.concat([pd.DataFrame.from_dict(out).drop("simplex_counts", axis=1), 
             pd.DataFrame.from_dict(out['simplex_counts'], orient="index").fillna(0).rename(columns={i:f"{i}_simplices" for i in range(4)})],
             axis=1)
    props['density']=(props['1_simplices']/
                      ((props["0_simplices"]*props["0_simplices"]-props["0_simplices"])/2))
    props['rc_over_nodes']=props['rc_edges']/props['0_simplices']
    props['rc_over_edges']=props['rc_edges']/props['1_simplices']
    return props

def compute_basics_over_nbds_ori_and_controls(cfg):
    '''Computes, nodes, edges, rc_edges and simplices up to dimension 3 for original and requested controls'''
    conn=load_connectome(cfg['connectome']['data_dir'], cfg['connectome']['name'])
    M=conn.matrix.astype('bool').tocsr() # remove potential weights of synapses
    print(f"The connectome has {M.diagonal().sum()} entries out of {M.sum()} in the diagonal; these will be ignored.")
    M.setdiag(0)
    M.eliminate_zeros()
    # Original
    compute_basics_over_neighborhoods(M).to_pickle(f"{cfg['save_path']}/{cfg['connectome']['name']}_nbd_basics_original.pkl")
    # Controls 
    seed=cfg["controls"]["seed"]
    for ctr_type in cfg["controls"]["types"]:
        f_ctr=getattr(randomization, ctr_type)
        if ctr_type!='run_DD2':
            adj=f_ctr(M, seed=seed)
        else:
            f_kw=cfg["controls"]["types"][ctr_type]['kwargs']
            xyz=conn.vertices[f_kw['xyz_labels']].to_numpy()
            n=len(conn.vertices)
            adj=f_ctr(n,f_kw['a'],f_kw['b'],xyz=xyz, seed=(seed, seed))
        compute_basics_over_neighborhoods(adj).to_pickle(f"{cfg['save_path']}/{cfg['connectome']['name']}_nbd_basics_{ctr_type}.pkl")


