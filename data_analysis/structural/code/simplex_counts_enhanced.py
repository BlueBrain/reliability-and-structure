# SPDX-FileCopyrightText: 2024 Blue Brain Project / EPFL
#
# SPDX-License-Identifier: AGPL-3.0-or-later

'''
Computing simplex counts and edge participation for manipulated connectomes
Author(s): Daniela Egas S. 
Last modified: 12.2023
'''
import conntility
from connalysis.network import topology 
import pandas as pd

def compute_and_save(path, name, root):
    print(f"Computing {name}")
    M=conntility.ConnectivityMatrix.from_h5(path).index('synapse_class').isin('EXC').matrix.tocsr().astype('bool')
    edge_par, sc=topology.edge_participation(M, threads=20, return_simplex_counts=True)
    sc=pd.Series(sc, index=pd.Index(range(len(sc)), name="dim"))
    edge_par.to_pickle(f'{root}edge_par_EE_{name}.pkl')
    sc.to_pickle(f'{root}sc_EE_{name}.pkl')
    print("Done\n")
    
def main(): 
    path_base="/gpfs/bbp.cscs.ch/data/scratch/proj9/bisimplices/simulations/BlobStimReliability_O1v5-SONATA_Baseline/working_dir/connectome.h5"
    root="/gpfs/bbp.cscs.ch/project/proj102/egas/reliability/manipulation_selection/data/"
    ns=[4, 20, 66, 100, 200, 300, 400, 500, 270, 470, 670]
    names=[f'V5_enhanced_{n}k' for n in ns]

    #Compute simplices and edge participation 
    #name='baseline'
    #compute_and_save(path_base, name,root)
    for name in names:
        compute_and_save(f"{root}{name}.h5", name,root)

if __name__=="__main__": 
    main()