# SPDX-FileCopyrightText: 2024 Blue Brain Project / EPFL
#
# SPDX-License-Identifier: AGPL-3.0-or-later

'''Functions to select simulations, compute reliability and bootstrap samples'''
import numpy as np
import tqdm
import os
import time
import itertools
import h5py
import math
import random
from scipy.spatial import distance


# Basic functions for reliability computations

def mean_center_spike_signals(spike_signals):
    spike_signals = spike_signals - np.mean(spike_signals, 2, keepdims=True)
    return spike_signals
    
def avg_reliability(v_filt, mean_center=True):
    """Computes average reliability between all pairs of trials of a give set.
        v_filt: Array spike trains many simuations of shape N_trials x #neuronss # time_bins"""
    no_cells=v_filt.shape[1]
    if mean_center:
        v_filt=mean_center_spike_signals(v_filt)
    mean_rels=[(1-distance.pdist(v_filt[:,x,:],'cosine')).mean() for x in range(no_cells)]
    return np.array(mean_rels)

# Basic reliability computations from single set of samples
def load_spike_signals(file, sim_idx,return_metadata=False):
    spike_signals = []
    with h5py.File(file, 'r') as f:
        gids = f['gids'][()]
        metadata={'firing_rates':f['firing_rates'][()],
                  #'mean_centered':f['mean_centered'][()], # Removed from later simulations because we don't mean center anymore
                  'sigma':f['sigma'][()]}
        for sim_id in sim_idx:
            spike_signals.append(f['spike_signals_exc'][f'sim_{sim_id}'][()])
    if return_metadata==False:
        return gids, np.stack(spike_signals)
    elif return_metadata==True:
        return gids, np.stack(spike_signals), metadata

def compute_reliabity_basic(save_path,spikes_h5_file, selected_sims_index, mean_center=True):
    ### Load seeds of single simulation/experiment and compute average reliability

    # Load data
    N = selected_sims_index.shape[0]
    print("Loading data")
    start=time.time()
    gids, spike_signals,metadata = load_spike_signals(spikes_h5_file, selected_sims_index,return_metadata=True)
    assert N == spike_signals.shape[0], "Error in the number of simulations loaded"
    assert gids.size == spike_signals.shape[1], "Missmatch on the number of gids and spike signals"
    sigma=metadata['sigma'];
    mean_centered=metadata['mean_centered'];
    firing_rates=metadata['firing_rates']
    print(f'Data loaded and checked correct dimension in {time.time()-start:.2f} sec')

    # Compute reliability
    start=time.time()
    reliab = avg_reliability(spike_signals, mean_center=mean_center)
    print(f'Computed reliabilty in {time.time()-start:.2f} sec')
    
    np.savez(reliab_save_path, reliab=reliab, gids=gids,
             sigma=sigma, mean_centered=mean_centered,
             sel_idx=selected_sims_index, firing_rates=firing_rates[selected_sims_index])
            #TODO: Save instead the path to the firing rates?

def compute_reliabity_basic_many_sims(config_dict, mean_center=True):
    ''' Load samples and compute average reliability of several seeds and store them together in a dict 
        config_dict has keys: 
            'save_path': the path on which the results will be stored
            'name_of_sim': a dict for each simulation to be analyzed, with keys: 
                    'spikes_h5_file': the path to the (pre-processed spike train files) 
                    'selected_sims_index': an array indicating which seeds to use from all available seeds
        Returns: dict with keys 'name_of_sim', each containing, the reliabity results and the metadata of the input
    '''
    # Looping through simulations 
    for key in config_dict.keys():
        spikes_h5_file= f'{config_dict[key]["sim_dir"]}/working_dir/processed_data_store.h5'
        selected_sims_index=config_dict[key]['selected_sims_index']
        # Load data
        N = selected_sims_index.shape[0]  # Total number of seeds selected
        print("Loading data")
        start=time.time()
        gids, spike_signals = load_spike_signals(spikes_h5_file, selected_sims_index,return_metadata=False)
        assert N == spike_signals.shape[0], "Error in the number of simulations loaded"
        assert gids.size == spike_signals.shape[1], "Missmatch on the number of gids and spike signals"
        print(f'Data of {key} loaded and checked correct dimension in {time.time()-start:.2f} sec')
        # Compute reliability
        start=time.time()
        reliability = avg_reliability(spike_signals, mean_center=mean_center)
        print(f'Computed reliabilty of {key} in {time.time()-start:.2f} sec')
        # Save
        np.savez(f'{config_dict[key]["sim_dir"]}/working_dir/{config_dict[key]["out_fname"]}',
                 reliability=reliability, selected_sims_index=selected_sims_index) 


# Functions for indexing block desing simulations
def get_boolean_index_block_gids(block_table, gids, special_gids):
    special_gids_index = [np.where(gids == gid)[0][0] for gid in special_gids]  # Indices special_gids
    special_block_index = np.full(len(gids), -1)
    special_block_index[special_gids_index] = np.arange(
        block_table.shape[0])  # Index of given cell in block design, -1 if not selected
    return special_gids_index, special_block_index


def select_sims_for_cell(gid_index, special_block_index, block_table):
    bidx = special_block_index[gid_index]  # Index of cell_index in block design
    assert (0 <= bidx) & (bidx < block_table.shape[1]), 'ERROR in reindexing block design'
    sim_sel = np.where(block_table[bidx, :])[0]  # Simulations where cell is manipulated
    return sim_sel


# Bootstrapping reliability
def random_subset_of_combinations(iterable, R, k, seed=0):
    """Returns `R` random samples of
    `k` length combinations of in `iterable`"""
    max_combinations = math.comb(len(iterable), k)
    assert max_combinations >= R, 'There are not enough samples to subsample via bootstrap'
    np.random.seed(seed)
    # indexes = set(random.sample(range(max_combinations), R))#indices of sampled combinations
    indexes = set(np.random.choice(max_combinations, R, replace=False))  # indices of sampled combinations
    max_idx = max(indexes)
    selected_combinations = []  # list of selected combinations
    for i, combination in zip(itertools.count(), itertools.combinations(iterable, k)):
        if i in indexes:
            selected_combinations.append(combination)
        elif i > max_idx:
            break
    return selected_combinations



def compute_and_save_reliabity_bootstrap(save_path,
                                         spikes_h5_file, selected_sims_index,
                                         R=100, k=10, mean_center=True,
                                         bootstrap_seed=0, force_recomp=False):
    ###Output directory
    # reliab_path: directory where to save the reliability files
    ### Input data
    # spikes_h5_file: h5_store containing spike signals and gids
    # selected_sims_index: Indices of the simulations selected for analysis
    # TODO: Would be good to have the rates and the seeds in the file above as it was before changing the format
    # TODO: FIRST CHECK IF FILES ARE THERE AND THEN SLICE, NOT THE OTHER WAY AROUND ...
    ### Meta data of the spike_signals
    # sigma : sigma for Gaussian kernel in pre-processing
    # mean_center: True if pre-processed activity per cell is mean centered
    # firing_rates: average firing rates related to the computation
    ### Input parameters
    # R: Bootstrap repetitions
    # k: Number of seeds to choose for each single computation
    # bootstrap_seed: seed of bootstrap sample
    # force_recomp = True # Force recomputation, even if file already exists

    # Load data
    N = selected_sims_index.shape[0]  # Total number of seeds for the bootstrap
    print("Loading data")
    start=time.time()
    gids, spike_signals,metadata = load_spike_signals(spikes_h5_file, selected_sims_index,return_metadata=True)
    assert N == spike_signals.shape[0], "Error in the number of simulations loaded"
    assert gids.size == spike_signals.shape[1], "Missmatch on the number of gids and spike signals"
    sigma=metadata['sigma'];
    mean_centered=metadata['mean_centered'];
    firing_rates=metadata['firing_rates']
    print(f'Data loaded and checked correct dimension in {time.time()-start:.2f} sec')
    # Select bootstrap samples, compute reliabilty and save
    print("Generating combinations for the bootstrap")
    bootstrap_combinations = random_subset_of_combinations(range(N), R, k, seed=bootstrap_seed)
    print("Done")
    for ridx in tqdm.tqdm(range(R)):
        reliab_save_path = os.path.join(save_path, f'reliability_{ridx:03d}.npz')
        if (os.path.exists(reliab_save_path) and (force_recomp==False)):
            print("Selection already computed")
        elif os.path.exists(reliab_save_path) or force_recomp:
            sel_idx = bootstrap_combinations[ridx]
            print("\nSlicing data")
            start=time.time()
            signals=spike_signals[sel_idx, :, :]  # Restrict to the k selected simulations
            print(f'Done slicing in {time.time()-start:.2f} sec')
            start=time.time()
            reliab = avg_reliability(signals, mean_center=mean_center)
            print(f'Reliability for slice computed in in {time.time()-start:.2f} sec')
            # print(reliab.shape, reliab.min(),reliab.max())
            np.savez(reliab_save_path, reliab=reliab, gids=gids,
                     sigma=sigma, mean_centered=mean_centered,
                     sel_idx=selected_sims_index, firing_rates=firing_rates[selected_sims_index])
            #TODO: We might also want to store sim_seeds and nrn_type
