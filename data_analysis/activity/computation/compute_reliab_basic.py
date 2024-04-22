# SPDX-FileCopyrightText: 2024 Blue Brain Project / EPFL
#
# SPDX-License-Identifier: AGPL-3.0-or-later

''' 
Compute reliability of connectomes with added connections
Author(s): Daniela Egas S. 
Last update: 12.2023 
'''

import numpy as np
import sys
sys.path.append('/gpfs/bbp.cscs.ch/home/egassant/reliability_and_structure/library')
from reliability import *

# List of all model names and their correponding directories
root='/gpfs/bbp.cscs.ch/data/scratch/proj9/bisimplices/simulations/'
# Names
models_base=["Baseline"]
models_remove=["rc_rem_56",
               "rc_rem_56_456",
               "rc_rem_456",
               "rc_rem_56_ctr",
               "rc_rem_56_456_ctr",
               "rc_rem_456_ctr",
               "rc_rem_all_rc"]
models_distance=["order1",
                 "order2",
                 "order3",
                 "order4",
                 "order5"]
models_add=['rc_add_4',
            'rc_add_4_ctr',
            'rc_add_2',
            'rc_add_2_ctr']
models_enhanced=[f'{n}00k' for n in np.arange(5)+1]+[f'670k']
# Paths to simulations
sims_base=["BlobStimReliability_O1v5-SONATA_Baseline"]
sims_remove=["BlobStimReliability_O1v5-SONATA_RecipRemoval_StructDim56",
             "BlobStimReliability_O1v5-SONATA_RecipRemoval_StructDim56_456",
             "BlobStimReliability_O1v5-SONATA_RecipRemoval_StructDim456",
             "BlobStimReliability_O1v5-SONATA_RecipRemoval_Unstruct-0",
             "BlobStimReliability_O1v5-SONATA_RecipRemoval_Unstruct-1",
             "BlobStimReliability_O1v5-SONATA_RecipRemoval_Unstruct-2",
             "BlobStimReliability_O1v5-SONATA_RecipRemoval_Unstruct-3"]
sims_distance=["BlobStimReliability_O1v5-SONATA_ConnRewired_mc2EE_Order1",
               "BlobStimReliability_O1v5-SONATA_ConnRewired_mc2EE_Order2",
               "BlobStimReliability_O1v5-SONATA_ConnRewired_mc2EE_Order3",
               "BlobStimReliability_O1v5-SONATA_ConnRewired_mc2EE_Order4",
               "BlobStimReliability_O1v5-SONATA_ConnRewired_mc2EE_Order5"]
sims_add=['BlobStimReliability_O1v5-SONATA_ConnAdd_RecipStruct0x4',
             'BlobStimReliability_O1v5-SONATA_ConnAdd_Control0x4',
             'BlobStimReliability_O1v5-SONATA_ConnAdd_RecipStruct0x2',
             'BlobStimReliability_O1v5-SONATA_ConnAdd_Control0x2']
sims_enhanced=[f'BlobStimReliability_O1v5-SONATA_ConnRewireEnhanced{n}K' for n in 
               [100, 200, 300, 400, 500,670]] 


def main():
    names= models_base + models_remove + models_distance + models_add + models_enhanced
    paths = sims_base + sims_remove + sims_distance + sims_add + sims_enhanced
    selected_seeds=np.arange(10)
    # Aggregate selcition in config dict 
    config_dict={names[i]:{
        "sim_dir":f"{root}{paths[i]}",
        "selected_sims_index":selected_seeds,
        "out_fname":"reliability_basic.npz"}
                 for i in range (len(names))
                }
    compute_reliabity_basic_many_sims(config_dict)


if __name__ == '__main__': 
    main()