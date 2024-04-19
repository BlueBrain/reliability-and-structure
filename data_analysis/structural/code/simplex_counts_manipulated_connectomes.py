''' 
Compute simplex counts of the full and E-E subgraphs of the original and manipulated connectomes. 
Author(s): Daniela Egas S. 
Last update: 11.2023 
'''

import scipy.sparse as sp
import pandas as pd
import conntility
from connalysis.network import topology 
root='/gpfs/bbp.cscs.ch/data/scratch/proj9/bisimplices/simulations/'

# Names of simulations of baseline and manipulated connectomes
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
sims_add=['BlobStimReliability_O1v5-SONATA_ConnAdd_Control0x2',
          'BlobStimReliability_O1v5-SONATA_ConnAdd_Control0x3',
          'BlobStimReliability_O1v5-SONATA_ConnAdd_Control0x4',
          'BlobStimReliability_O1v5-SONATA_ConnAdd_Control0x5', 
          'BlobStimReliability_O1v5-SONATA_ConnAdd_Control0x8',
          'BlobStimReliability_O1v5-SONATA_ConnAdd_Control0x16',
          'BlobStimReliability_O1v5-SONATA_ConnAdd_RecipStruct0x2',
          'BlobStimReliability_O1v5-SONATA_ConnAdd_RecipStruct0x3',
          'BlobStimReliability_O1v5-SONATA_ConnAdd_RecipStruct0x4',
          'BlobStimReliability_O1v5-SONATA_ConnAdd_RecipStruct0x5',
          'BlobStimReliability_O1v5-SONATA_ConnAdd_RecipStruct0x8',
          'BlobStimReliability_O1v5-SONATA_ConnAdd_RecipStruct0x16']

#for sim_name in sims_base+sims_distance+sims_remove+sims_add:
import numpy as np
for sim_name in np.array(sims_add)[[1, 3, 7, 9]]:
    print(f'Starting {sim_name}')
    path=f'{root}{sim_name}/working_dir/'
    connectome=conntility.ConnectivityMatrix(sp.load_npz(f'{path}connectivity.npz'), 
                                             vertex_properties=pd.read_pickle(f'{path}neuron_info.pickle'))
    sc_exc=topology.simplex_counts(connectome.index('synapse_class').isin('EXC').matrix.tocsr())
    sc_full=topology.simplex_counts(connectome.matrix.tocsr())
    df=pd.concat([sc_exc, sc_full], axis=1) 
    df.columns=["simplex_counts_EXC", "simplex_counts_full"]
    df.to_pickle(f'{path}simplex_counts.pkl')
    print(f'Done with {sim_name}')