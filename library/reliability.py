'''Functions to select simulations, compute reliability and bootstrap samples'''
import numpy as np
import tqdm
import os


# Functions for reliability computations
def reliability(v1, v2):
    """v1/v2: spike signals of two simulations/set of neurons.  Arrays of size # neurons x # time_bins"""
    product_of_norms = (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))
    mask_to_normalize_non_zero = (product_of_norms != 0)
    reliability = (v1 * v2).sum(axis=1)
    reliability[mask_to_normalize_non_zero] = reliability[mask_to_normalize_non_zero] / product_of_norms[
        mask_to_normalize_non_zero]
    return reliability


def avg_reliability(v_filt):
    """Computes average reliability between all pairs of trials of a give set.
    v_filt: Array spike trains many simuations of shape N_trials x #neuronss # time_bins"""
    N_trials = v_filt.shape[0]
    avg_rel = np.zeros(v_filt.shape[1])
    for i, j in itertools.combinations(range(N_trials), 2):
        avg_rel = avg_rel + reliability(v_filt[i, :, :], v_filt[j, :, :])
    avg_rel = 2 * avg_rel / (N_trials * (N_trials - 1))
    return avg_rel


# Functions for indexing block desing and loading selected simulations
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


def load_spike_signals(file, sim_idx):
    import h5py
    spike_signals = []
    with h5py.File(file, 'r') as f:
        gids = f['gids'][()]
        for sim_id in sim_idx:
            spike_signals.append(f['spike_signals_exc'][f'sim_{sim_id}'][()])
    return gids, np.stack(spike_signals)


# Reliability bootstrap
def random_subset_of_combinations(iterable, R, k, seed=0):
    """Returns `R` random samples of
    `k` length combinations of in `iterable`"""
    import math
    import random
    import itertools
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
                                         sim_seeds, sigma, mean_centered, nrn_type,
                                         R=100, k=10,
                                         bootstrap_seed=0, force_recomp=False):
    ###Output directory
    # reliab_path: directory where to save the reliability files
    ### Input data
    # spikes_h5_file: h5_store containing spike signals and gids
    # selected_sims_index: Indices of the simulations selected for analysis
    # TODO: Would be good to have the rates and the seeds in the file above as it was before changing the format
    ### Meta data of the spike_signals
    # sim_seeds: seeds of the simulations in h5_store
    # sigma : sigma for Gaussian kernel in pre-processing
    # mean_center: True if pre-processed activity per cell is mean centered
    # nrn_type: neuron type, generically EXC or INH
    ### Input parameters
    # R: Bootstrap repetitions
    # k: Number of seeds to choose for each single computation
    # bootstrap_seed: seed of bootstrap sample
    # force_recomp = True # Force recomputation, even if file already exists

    # Load data
    N = selected_sims_index.shape[0]  # Total number of seeds for the bootstrap
    gids, spike_signals = load_spike_signals(spikes_h5_file, selected_sims_index)
    assert N == spike_signals.shape[0], "Error in the number of simulations loaded"
    assert gids.size == spike_signals.shape[1], "Missmatch on the number of gids and spike signals"
    # Select bootstrap samples, compute reliabilty and save
    bootstrap_combinations = random_subset_of_combinations(range(N), R, k, seed=bootstrap_seed)
    print('Loaded files and selected combinations for bootstrap')
    for ridx in tqdm.tqdm(range(R)):
        sel_idx = bootstrap_combinations[ridx]
        spike_signals[sel_idx, :, :]  # Restrict to the k selected simulations
        # Paths for saving
        reliab_save_path = os.path.join(save_path, f'reliability_{ridx:03d}.npz')
        if not os.path.exists(reliab_save_path) or force_recomp:
            reliab = avg_reliability(spike_signals)
            # print(reliab.shape, reliab.min(),reliab.max())
            np.savez(reliab_save_path, reliab=reliab, gids=gids,
                     nrn_type=nrn_type, sigma=sigma, mean_centered=mean_centered,
                     sel_idx=selected_sims_index, sim_seeds=sim_seeds[selected_sims_index])
            # reliab_paths=reliab_paths #Daniela: I'm not sure what this is suppossed to be so I commented it out'''
        '''Old code for saving rates.  Format to new format and add to file. 
        rates_file = os.path.join(save_path, folder_name, f'rates{rates_type}_{ridx:03d}.npz')
        if not os.path.exists(rates_file) or force_recomp:
            np.savez(rates_file, rates=rates[:, sel_idx], gids=gids, reliab_paths=reliab_paths, nrn_type=nrn_type,
                     rates_type=rates_type, sel_idx=sel_idx, sim_seeds=sim_seeds[sel_idx])'''

