# SPDX-FileCopyrightText: 2024 Blue Brain Project / EPFL
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import sys 
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

# Network specific
from connalysis.network import topology
from connalysis import randomization

# Project specific
from structural_basic import load_connectome

def add_rc_on_simplices(cfg):
    # For saving 
    root_out=cfg["save"]["dir"]
    # Load base matrix
    if "project_load" in cfg["connectome_params"]: 
        # Load one of the 4 pre-specified connectomes in this project (BBP, MICrONS, Drosophila, Celegans) 
        conn_mat=load_connectome(cfg["connectome_params"]["project_load"]["connectome_dir"],
                                 cfg["connectome_params"]["project_load"]["conn_type"])
    else:
        # Load a connectome from a given directory
        conn_mat=conntility.ConnectivityMatrix.from_h5(cfg["direct_load"])

    # Get edge participation 
    A=conn_mat.matrix.tocsr().astype(bool)
    print(f"The original matrix has {A.diagonal().sum()} entries in the diagonal and these will be ignored")
    A.setdiag(0)
    A.eliminate_zeros()
    if cfg["edge_par"]["precomputed"]:
        edge_par_ori=pd.read_pickle(cfg["edge_par"]["path_edge_par"])
        
    else:
        edge_par_ori, counts=topology.edge_participation(A, max_simplices=cfg["edge_par"]["max_simplices"], 
                                                 threads=cfg["edge_par"]["threads"], return_simplex_counts=True)

    # Compute skeleta of original matrix 
    start=time.time()
    N=A.shape[0]
    print(f"Size of matrix {N}")
    skeleta=topology.get_k_skeleta_graph(A, N=N,edge_par=edge_par_ori)
    dimensions=np.arange(2,edge_par_ori.columns.stop)
    print(f"Dimensions to manipulate {dimensions}")
    print(f"Time to compute modified graph {(time.time()-start)/60:.3f} minutes")

    # Add rc on skeleta 
    mats={"original":A} 
    if cfg["mod_params"]["compute_simplices"]:
        if cfg["edge_par"]["precomputed"]:
            assert isinstance(cfg["mod_params"]["path_simp_counts_original"], str), "A path to simplex counts needs to be provided"
            assert os.path.isfile(cfg["mod_params"]["path_simp_counts_original"]),  "The file with simplex counts must exist"
            counts=np.load(cfg["mod_params"]["path_simp_counts_original"]) 
        sc={'original':counts}
        
    for factors in tqdm(cfg["mod_params"]["blowups"]):
        start=time.time()
        mats[f'modified_{factors}']={}; #sc[f'modified_{factors}']={};
        if cfg["mod_params"]["compute_simplices"]:
                sc[f'modified_{factors}']={}
        for seed in cfg["mod_params"]["seeds"]:
            mats[f'modified_{factors}'][seed]=randomization.add_rc_connections_skeleta(mats['original'],factors=factors,
                                                                                       dimensions=dimensions,skeleta=skeleta,
                                                                                       seed=seed)
            if cfg["mod_params"]["compute_simplices"]:
                sc[f'modified_{factors}'][seed]=topology.simplex_counts(mats[f'modified_{factors}'][seed], threads=8)
        print(f"Done with blowup factor {factors} in {(time.time()-start)/60:.3f} minutes")

    if cfg["mod_params"]["compute_controls"]:
        control_mats={key:{} for key in mats}; control_sc={key:{} for key in mats}
        for factor in tqdm(cfg["mod_params"]["blowups"]):
            for seed in cfg["mod_params"]["seeds"]:
                extra_edges=mats[f'modified_{factor}'][seed].nnz-mats['original'].nnz
                control_mats[f'modified_{factor}'][seed]=randomization.add_connections(mats['original'],extra_edges, seed=seed)
                if cfg["mod_params"]["compute_simplices"]:
                    control_sc[f'modified_{factor}'][seed]=topology.simplex_counts(control_mats[f'modified_{factor}'][seed], threads=10)
    
    # Saving results 
    path_out=f'{root_out}/{cfg["save"]["preffix"]}{cfg["save"]["fname_mats"]}'
    with open(path_out, 'wb') as f:
        pickle.dump(mats, f)
    if cfg["mod_params"]["compute_controls"]:
        path_out=f'{root_out}/{cfg["save"]["preffix"]}{cfg["save"]["fname_mats_ctr"]}'
        with open(path_out, 'wb') as f:
            pickle.dump(control_mats, f)
    if cfg["mod_params"]["compute_simplices"]:
        path_out=f'{root_out}/{cfg["save"]["preffix"]}{cfg["save"]["fname_sc"]}'
        with open(path_out, 'wb') as f:
            pickle.dump(sc, f)
        if cfg["mod_params"]["compute_controls"]:
            path_out=f'{root_out}/{cfg["save"]["preffix"]}{cfg["save"]["fname_sc_ctr"]}'
            with open(path_out, 'wb') as f:
                pickle.dump(control_sc, f)
    print(f"Results saved in {root_out}")
