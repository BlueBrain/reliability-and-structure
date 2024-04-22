# SPDX-FileCopyrightText: 2024 Blue Brain Project / EPFL
#
# SPDX-License-Identifier: AGPL-3.0-or-later

''' 
Compute reliability bootstrap of a given set of sims 
Author(s): Daniela Egas S. 
Last update: 10.2023 
'''

import os
import sys
import numpy as np
sys.path.append('/gpfs/bbp.cscs.ch/home/egassant/reliability_and_structure/library')
import reliability
import time
import pandas as pd

root='/gpfs/bbp.cscs.ch/data/scratch/proj9/bisimplices/simulations/'
sim_name = sys.argv[1]

start=time.time()
# Data loaded from h5_store
sim_path = os.path.join(root, sim_name)
spikes_h5_file=f'{root}/{sim_name}/working_dir/processed_data_store.h5'
# Output files 
save_path=f'{root}/{sim_name}/working_dir/'
# Use all 30 simulations
selected_sims_index=np.arange(30)
# Parameters for the bootstrap
R=300; k=10; force_recomp=True
# Compute reliability
reliability.compute_and_save_reliabity_bootstrap(save_path=save_path,
                                     spikes_h5_file=spikes_h5_file, selected_sims_index=selected_sims_index,
                                     R=R, k=k, force_recomp=force_recomp)  
print(f'Done in {(time.time()-start)/(60*60):.2f} hours') 

# Aggregating data 
rels={}
for ridx in np.arange(0,R):
    file_name = f'{save_path}reliability_{ridx:03d}.npz'
    rels[f'sample_{ridx}']=np.load(file_name)['reliab']
rels=pd.DataFrame.from_dict(rels)
rels.insert(0, 'gid',np.load(file_name)['gids'] ) 
rels.to_pickle(f'{save_path}reliability_bootstrap.pkl')