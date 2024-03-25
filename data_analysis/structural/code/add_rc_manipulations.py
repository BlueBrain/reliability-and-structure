''' 
Generate matrices with added reciprocal connections along maximal simplices and corresponding controls
Author(s): Daniela Egas S. 
Last update: 03.2024
TODO: Add zenodo link 
'''

import sys 
sys.path.append('../../../library')
from rc_manipulations import add_rc_on_simplices


cfg={
    "connectome_params":{ #how to load the ConnectivityMatrix
        #"direct_load":"path"# Generic ConnectivityMatrix in this path
        "project_load":{# Specific to this project
            # In connectome_dir add the directory where the connectivity data has been downloade from zenodo 
            "connectome_dir":"/gpfs/bbp.cscs.ch/home/egassant/reliability_and_structure/data_analysis/data", 
            "conn_type":"BBP"}
    },
    
    "edge_par":{
        "precomputed": True, # True if edge_participation has been precomputed, in which case a path to it needs to be given below. 
                             # If false it will be computed on the fly 
        "path_edge_par":"../../data/BBP_edge_participation_maximal_E_E.pkl", 
                        # path to precomputed edge participation
        "max_simplices":True, # If true edge participation will be computed in maximal simplices
        "threads":10
    },
    "mod_params":{ # parameters of modification
        "seeds":[0,1,2,3,4,5,6,7,8,9], 
        "blowups":[2,4, 8, 16], # factor by which the number of rc in simplices will be multiplied by
        "compute_controls": True, # compute matrices where the same number or rc are added at random
        "compute_simplices": True, # count simplices in the generated matrices
        "path_simp_counts_original":"../../data/BBP_sc_E_E.npy"
                                    # path to simplex counts in the original graph, 
                                    # needs to be provide if edge participation is precomputed

    },
    "save":{
        "dir":"../../data", #directory to store results
        "preffix":"BBP",
        "fname_mats":"mats_rc_on_simplices.pkl",
        "fname_mats_ctr":"mats_rc_on_simplices_controls.pkl",
        "fname_sc":"sc_rc_on_simplices.pkl",
        "fname_sc_ctr":"sc_rc_on_simplices_controls.pkl"
    }
        
    
}



if __name__ == "__main__":
    add_rc_on_simplices(cfg) 