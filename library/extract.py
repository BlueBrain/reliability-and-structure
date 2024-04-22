# SPDX-FileCopyrightText: 2024 Blue Brain Project / EPFL
#
# SPDX-License-Identifier: AGPL-3.0-or-later

'''Description: Functions to extract (and cut) spike trains in topo-sample format
   Author: C. Pokorny
   Date: 11/2022
'''

import json
import numpy as np
import os
import scipy.sparse as sps
from bluepy import Cell, Circuit, Simulation


def run_extraction(campaign_path, working_dir_name='working_dir', cell_target='mc2_Column', syn_class='EXC'):
    """ Run extraction of neuron info, stimulus train, adjacency matrix, and EXC/INH/ALL spike trains. """

    # Load campaign config
    with open(os.path.join(campaign_path, 'config.json'), 'r') as f:
        config_dict = json.load(f)
    sim_paths = [os.path.join(config_dict['attrs']['path_prefix'], p) for p in config_dict['data']]

    # Set up save path (= working dir)
    save_path = os.path.join(config_dict['attrs']['path_prefix'], config_dict['name'], working_dir_name)
    if not os.path.exists(save_path): # Create results folder
        os.makedirs(save_path)
        os.symlink(save_path, os.path.join(campaign_path, os.path.split(save_path)[-1])) # Create symbolic link

    # Get stimulus config
    blue_config = os.path.join(sim_paths[0], 'BlueConfig')
    sim = Simulation(blue_config)
    c = sim.circuit
    target_gids = c.cells.ids(cell_target)
    input_spike_file = os.path.abspath(os.path.join(sim.config['Run_Default']['OutputRoot'], sim.config['Stimulus_spikeReplay']['SpikeFile']))
    stim_config_file = os.path.splitext(input_spike_file)[0] + '.json'

    # Extract neuron info [same for all sims]
    neuron_info = extract_neuron_info(c, target_gids, save_path)

    # Extract adjacency matrix [same for all sims]
    _ = extract_adj_matrix(c, neuron_info, save_path)

    # Extract stim train & time windows [assumed to be same for all sims; will be check below]
    stim_train, time_windows = extract_stim_train(stim_config_file, save_path)
    cut_start = np.min(time_windows)
    cut_end = np.max(time_windows)
    time_windows_cut = time_windows[np.logical_and(time_windows >= cut_start, time_windows <= cut_end)] - cut_start
    np.save(os.path.join(save_path, 'time_windows.npy'), time_windows_cut)

    # Check & extract spike trains
    for sidx, sim_path in enumerate(sim_paths):

        # Check stim train
        sim_tmp = Simulation(os.path.join(sim_path, 'BlueConfig'))
        spike_file_tmp = os.path.abspath(os.path.join(sim_tmp.config['Run_Default']['OutputRoot'], sim_tmp.config['Stimulus_spikeReplay']['SpikeFile']))
        stim_config_file_tmp = os.path.splitext(spike_file_tmp)[0] + '.json'
        stim_train_tmp, time_windows_tmp = extract_stim_train(stim_config_file_tmp)
        assert np.array_equal(stim_train, stim_train_tmp), 'ERROR: Stim train mismatch!'
        assert np.array_equal(time_windows, time_windows_tmp), 'ERROR: Time windows mismatch!'

        # Extract excitatory/inhibitory/all spikes
        _ = extract_spikes(sim_tmp, neuron_info, syn_class, cut_start, cut_end, save_path, fn_spec=str(sidx))

    print(f'INFO: {sidx + 1} spike files written to "{save_path}"')

    return sim_paths, save_path


def extract_neuron_info(circuit, gids, save_path=None):
    neuron_info = circuit.cells.get(gids, properties=[Cell.X, Cell.Y, Cell.Z, Cell.LAYER, Cell.MTYPE, Cell.SYNAPSE_CLASS])

    if save_path is not None:
        neuron_info.to_pickle(os.path.join(save_path, 'neuron_info.pickle'))
        neuron_info.to_hdf(os.path.join(save_path, 'neuron_info.h5'), 'neuron_info', format='table')

    return neuron_info


def extract_spikes(sim, neuron_info, syn_class=None, cut_start=None, cut_end=None, save_path=None, fn_spec=None):
    if syn_class is None:
        syn_class = 'ALL'
    if syn_class == 'ALL':
        gids = neuron_info.index
    else:
        gids = neuron_info[neuron_info['synapse_class'] == syn_class].index

    if cut_start is None:
        cut_start = 0
    if cut_end is None:
        cut_end = np.inf

    if fn_spec is None:
        fn_spec = ''
    if len(fn_spec) > 0:
        fn_spec = f'_{fn_spec}'

    raw_spikes = sim.spikes.get(gids)
    raw_spikes = np.vstack((raw_spikes.index, raw_spikes.to_numpy())).T
    raw_spikes = raw_spikes[np.logical_and(raw_spikes[:, 0] >= cut_start, raw_spikes[:, 0] < cut_end), :] # Cut spikes
    raw_spikes[:, 0] = raw_spikes[:, 0] - cut_start # Correct spike times

    if save_path is not None:
        np.save(os.path.join(save_path, f'raw_spikes_{syn_class.lower()}{fn_spec}.npy'), raw_spikes)

    return raw_spikes


def extract_stim_train(stim_config_file, save_path=None):
    assert os.path.exists(stim_config_file), 'ERROR: Stimulus config file not found!'
    with open(stim_config_file, 'r') as f:
        stim_config = json.load(f)

    stim_stream = np.array(stim_config['props']['stim_train'])
    time_windows = np.array(stim_config['props']['time_windows'])

    if save_path is not None:
        np.save(os.path.join(save_path, 'stim_stream.npy'), stim_stream)

    return stim_stream, time_windows


def extract_adj_matrix(circuit, neuron_info, save_path=None):
    gids = neuron_info.index
    conns = np.array(list(circuit.connectome.iter_connections(pre=gids, post=gids)))
    reindex_table = sps.csr_matrix((np.arange(neuron_info.shape[0], dtype=int), (np.zeros(neuron_info.shape[0], dtype=int), neuron_info.index.to_numpy())))
    conns_reindex = np.array([reindex_table[0, conns[:, d]].toarray().flatten() for d in range(conns.shape[1])]).T

    adj_matrix = sps.csc_matrix((np.full(conns_reindex.shape[0], True), conns_reindex.T.tolist()))

    if save_path is not None:
        sps.save_npz(os.path.join(save_path, 'connectivity.npz'), adj_matrix)

    return adj_matrix

