'''Description: Functions to extract (and cut) spike trains in topo-sample format
                using bluepysnap and SONATA circuit configs
   Author: C. Pokorny
   Date: 09/2023
'''

import json
import numpy as np
import os
import pandas as pd
import scipy.sparse as sps
from bluepysnap import Circuit, Simulation


def run_extraction(campaign_path, nodes_popul_name=None, edges_popul_name=None, working_dir_name='working_dir', cell_target='hex0'):
    """ Run extraction of neuron info, stimulus train, adjacency matrix, and EXC spike trains. """

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
    sim0_path = sim_paths[0]
    sonata_config = os.path.join(sim0_path, 'simulation_config.json')
    sim = Simulation(sonata_config)
    c = sim.circuit
    assert nodes_popul_name in c.nodes.population_names, f'ERROR: Nodes population must be one of {c.nodes.population_names}!'
    if edges_popul_name is None:
        edges_popul_name = f'{nodes_popul_name}__{nodes_popul_name}__chemical'
        assert edges_popul_name in c.edges.population_names, f'ERROR: Edges population must be one of {c.edges.population_names}!'
    nodes = c.nodes[nodes_popul_name]
    edges = c.edges[edges_popul_name]
    target_nids = nodes.ids(cell_target)
    input_spike_file = os.path.abspath(os.path.join(sim0_path, sim.config['inputs'].get('Stimulus_spikeReplay', {}).get('spike_file', '')))
    stim_config_file = os.path.splitext(input_spike_file)[0] + '.json'
    if not os.path.exists(stim_config_file):  # Alternative stimulus file, like from PyramidBaseStimulus
        stim_stream_fn = f"stimulus_stream__start{config_dict['attrs']['stim_start']}__end{config_dict['attrs']['stim_end'] + 1}__rate{config_dict['attrs']['stim_rate']:.0f}__seed{config_dict['attrs']['stim_seed']}.txt"
        stim_config_file = os.path.join(os.path.split(sim0_path)[0], 'input_spikes', stim_stream_fn)

    # Extract neuron info [same for all sims]
    neuron_info = extract_neuron_info(nodes, target_nids, save_path)

    # Extract adjacency matrix [same for all sims]
    _ = extract_adj_matrix(edges, neuron_info, save_path)

    # Extract stim train & time windows [assumed to be same for all sims; will be check below]
    stim_train, time_windows = extract_stim_train(stim_config_file, save_path)
    cut_start = np.min(time_windows)
    cut_end = np.max(time_windows)
    time_windows_cut = time_windows[np.logical_and(time_windows >= cut_start, time_windows <= cut_end)] - cut_start
    np.save(os.path.join(save_path, 'time_windows.npy'), time_windows_cut)

    # Check & extract spike trains
    for sidx, sim_path in enumerate(sim_paths):

        # Check stim train
        sim_tmp = Simulation(os.path.join(sim_path, 'simulation_config.json'))
        
        spike_file_tmp = os.path.abspath(os.path.join(sim_path, sim_tmp.config['inputs'].get('Stimulus_spikeReplay', {}).get('spike_file', '')))
        stim_config_file_tmp = os.path.splitext(spike_file_tmp)[0] + '.json'
        if not os.path.exists(stim_config_file_tmp):  # Alternative stimulus file, like from PyramidBaseStimulus
            stim_stream_fn_tmp = f"stimulus_stream__start{config_dict['attrs']['stim_start']}__end{config_dict['attrs']['stim_end'] + 1}__rate{config_dict['attrs']['stim_rate']:.0f}__seed{config_dict['attrs']['stim_seed']}.txt"
            stim_config_file_tmp = os.path.join(os.path.split(sim_path)[0], 'input_spikes', stim_stream_fn_tmp)
        stim_train_tmp, time_windows_tmp = extract_stim_train(stim_config_file_tmp)
        assert np.array_equal(stim_train, stim_train_tmp), 'ERROR: Stim train mismatch!'
        assert np.array_equal(time_windows, time_windows_tmp), 'ERROR: Time windows mismatch!'

        # Extract excitatory spikes
        _ = extract_spikes(sim_tmp.spikes[nodes_popul_name], neuron_info, 'EXC', cut_start, cut_end, save_path, fn_spec=str(sidx))

    print(f'INFO: {sidx + 1} spike files written to "{save_path}"')

    return sim_paths, save_path


def extract_neuron_info(nodes, nids, save_path=None):
    neuron_info = nodes.get(nids, properties=['x', 'y', 'z', 'layer', 'mtype', 'synapse_class'])

    if save_path is not None:
        neuron_info.to_pickle(os.path.join(save_path, 'neuron_info.pickle'))
        neuron_info.to_hdf(os.path.join(save_path, 'neuron_info.h5'), 'neuron_info', format='table')

    return neuron_info


def extract_spikes(spikes, neuron_info, syn_class=None, cut_start=None, cut_end=None, save_path=None, fn_spec=None):
    if syn_class is None:
        syn_class = 'ALL'
    if syn_class == 'ALL':
        nids = neuron_info.index
    else:
        nids = neuron_info[neuron_info['synapse_class'] == syn_class].index

    if cut_start is None:
        cut_start = 0
    if cut_end is None:
        cut_end = np.inf

    if fn_spec is None:
        fn_spec = ''
    if len(fn_spec) > 0:
        fn_spec = f'_{fn_spec}'

    raw_spikes = spikes.get(nids)
    raw_spikes = np.vstack((raw_spikes.index, raw_spikes.to_numpy())).T
    raw_spikes = raw_spikes[np.logical_and(raw_spikes[:, 0] >= cut_start, raw_spikes[:, 0] < cut_end), :] # Cut spikes
    raw_spikes[:, 0] = raw_spikes[:, 0] - cut_start # Correct spike times

    if save_path is not None:
        np.save(os.path.join(save_path, f'raw_spikes_{syn_class.lower()}{fn_spec}.npy'), raw_spikes)

    return raw_spikes


def extract_stim_train(stim_config_file, save_path=None):
    assert os.path.exists(stim_config_file), 'ERROR: Stimulus config file not found!'
    if os.path.splitext(stim_config_file) == 'json':  # Stimulus config file
        with open(stim_config_file, 'r') as f:
            stim_config = json.load(f)

        stim_stream = np.array(stim_config['props']['stim_train'])
        time_windows = np.array(stim_config['props']['time_windows'])
    else:  # Stim stream text/csv file
        def pat_to_idx(pat):
            idx = np.array([ord(_p) for _p in pat]) # Convert pattern strings 'A', 'B', ...
            idx = idx - ord('A')                    # to indices 0, 1, ...
            return idx
        stim_stream_df = pd.read_csv(stim_config_file, sep=' ', header=None, names=['Time', 'Pattern'])
        stim_stream = pat_to_idx(stim_stream_df['Pattern'])
        t_stim = stim_stream_df['Time'].to_numpy()
        isi = np.min(np.diff(t_stim))
        time_windows = np.hstack([t_stim, t_stim[-1] + isi])

    if save_path is not None:
        np.save(os.path.join(save_path, 'stim_stream.npy'), stim_stream)

    return stim_stream, time_windows


def extract_adj_matrix(edges, neuron_info, save_path=None):
    nids = neuron_info.index
    conns = np.array(list(edges.iter_connections(source=nids, target=nids)))
    reindex_table = sps.csr_matrix((np.arange(neuron_info.shape[0], dtype=int), (np.zeros(neuron_info.shape[0], dtype=int), neuron_info.index.to_numpy())))
    conns_reindex = np.array([reindex_table[0, conns[:, d]].toarray().flatten() for d in range(conns.shape[1])]).T

    adj_matrix = sps.csc_matrix((np.full(conns_reindex.shape[0], True), conns_reindex.T.tolist()))

    if save_path is not None:
        sps.save_npz(os.path.join(save_path, 'connectivity.npz'), adj_matrix)

    return adj_matrix

