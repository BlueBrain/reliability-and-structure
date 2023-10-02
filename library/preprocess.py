'''Description: Functions to pre-process spike trains via gaussian kernel and mean centering
   Author: C. Pokorny
   Created: 11/2022
   Last modified: 10/2023
'''

import h5py
import json
import multiprocessing
import numpy as np
import os
import pandas as pd
import shutil
import time
import tqdm
from scipy.ndimage import gaussian_filter1d


def run_preprocessing(working_dir, spike_file_names, sigma=10.0, pool_size=10):
    """ Run preprocessing (filtering, mean-centering) of EXC spike trains. [PARALLEL IMPLEMENTATION] """

    spike_files = [os.path.join(working_dir, fn) for fn in spike_file_names]

    neuron_info = pd.read_pickle(os.path.join(working_dir, 'neuron_info.pickle'))
    gids = neuron_info[neuron_info['synapse_class'] == 'EXC'].index # Excitatory GIDs extracted from neuron info table

    time_windows = np.load(os.path.join(working_dir, 'time_windows.npy'))
    t_max = np.max(time_windows)

    # Run preprocessing [PARALLEL IMPLEMENTATION]
    fct_args = [(f, gids, t_max, sigma) for f in spike_files]
    t0 = time.time()
    with multiprocessing.Pool(pool_size) as pool:
        pool.map(proc_fct, fct_args)
    print(f'Finished preprocessing in {time.time() - t0:.3f}s')


def proc_fct(fct_args):
    """ Processing function for parallel implementation. """
    spike_file, gids, t_max, sigma = fct_args

    fidx = int(os.path.splitext(spike_file)[0].split('_')[-1])
    spikes, _, _ = load_spike_trains(spike_file)
    spike_signals, t_bins = extract_binned_spike_signals(spikes, gids, t_max, bin_size=1.0, save_path=None, fn_spec=None)
    spike_signals = filter_spike_signals(spike_signals, gids, t_bins, sigma, save_path=None, fn_spec=None)

    save_path=os.path.split(spike_file)[0]
    fn_spec=f'_exc_{fidx}__tmp__'
    np.savez_compressed(os.path.join(save_path, f'spike_signals{fn_spec}.npz'), spike_signals=spike_signals, t_bins=t_bins, gids=gids, sigma=sigma)


def merge_into_h5_data_store(working_dir, processed_file_names, data_store_name, split_by_gid=False):
    """ Creates new .h5 data store and merges processed EXC spike signals into .h5 store as separate
        data sets sims and optionally, GIDs. [WITH COMPRESSION] """

    spike_files = [os.path.join(working_dir, fn) for fn in processed_file_names]

    # Check that all files exist
    for spike_file in spike_files:
        assert os.path.exists(spike_file), f'ERROR: Spike file "{spike_file}" missing!'

    # Check that .h5 store does not yet exists
    h5_file = os.path.join(working_dir, data_store_name + '.h5')
    assert not os.path.exists(h5_file), 'ERROR: h5 store already exists!'

    # Merge files into .h5 store [WITH COMPRESSION!]
    h5f = h5py.File(h5_file, 'w-')
    grp = h5f.create_group('spike_signals_exc')

    t_bins = None
    gids = None
    sigma = None
    for spike_file in tqdm.tqdm(spike_files):
        fidx = int(os.path.splitext(spike_file)[0].replace('__tmp__', '').split('_')[-1])
        sp_dict = np.load(spike_file)
        if t_bins is None:
            t_bins = sp_dict['t_bins']
        else:
            assert np.array_equal(t_bins, sp_dict['t_bins']), 'ERROR: t_bins mismatch!'
        if gids is None:
            gids = sp_dict['gids']
        else:
            assert np.array_equal(gids, sp_dict['gids']), 'ERROR: GIDs mismatch!'
        if sigma is None:
            sigma = sp_dict['sigma']
        else:
            assert sigma == sp_dict['sigma'], 'ERROR: Sigma mismatch!'
        spike_signals = sp_dict['spike_signals']
        assert spike_signals.shape[0] == len(gids), 'ERROR: Spike signal shape mismatch!'

        if split_by_gid: # Data stored per sim & GID, i.e., spike_signal = np.array(h5_store['spike_signals_exc'][f'sim_{<sim_idx>}'][f'gid_{<gid>}'])
            sim_grp = grp.create_group(f'sim_{fidx}')
            for gidx, gid in enumerate(gids):
                sim_grp.create_dataset(f'gid_{gid}', data=spike_signals[gidx, :], compression='gzip', compression_opts=9)
        else: # Data stored per sim only, i.e., spike_signals = np.array(h5_store['spike_signals_exc'][f'sim_{<sim_idx>}'])
            grp.create_dataset(f'sim_{fidx}', data=spike_signals, compression='gzip', compression_opts=9)

    h5f.create_dataset('t_bins', data=t_bins)
    h5f.create_dataset('gids', data=gids)
    h5f.create_dataset('sigma', data=sigma)
    h5f.close()

    print(f'INFO: {len(spike_files)} files merged into "{h5_file}"')

    return h5_file


def load_spike_trains(spike_file):
    spikes = np.load(spike_file, allow_pickle=True)
    gids = np.unique(spikes[:, 1]).astype(int)
    t_max = np.max(spikes[:, 0]) # (ms)

    # print(f'INFO: Loaded {spikes.shape[0]} spikes from {len(gids)} GIDs (t_max={t_max}ms)')

    return spikes, gids, t_max


def extract_binned_spike_signals(spikes, gids, t_max, bin_size=1.0, save_path=None, fn_spec=None):
    # t_max ... Max. time in ms
    # bin_size ... Time resolution in ms

    assert np.all(np.isin(np.unique(spikes[:, 1]), gids)), 'ERROR: GID mismatch!'
    assert np.max(spikes[:, 0]) <= t_max, 'ERROR: Spike time exceeds max. time!'

    gid_bins = np.hstack([gids - 0.5, np.max(gids) + 0.5])
    t_bins = np.arange(0.0, t_max + bin_size, bin_size)

    ### [SLOW] spike_signals = np.array([np.histogram(spikes[spikes[:, 1] == gid, 0], bins=t_bins)[0] for gid in gids]).astype(float)
    spike_signals = np.histogram2d(spikes[:, 1], spikes[:, 0], bins=(gid_bins, t_bins))[0] # Andras Ecker's verions from https://github.com/andrisecker/assemblyfire

    if fn_spec is None:
        fn_spec = ''
    if len(fn_spec) > 0:
        fn_spec = f'_{fn_spec}'

    if save_path is not None:
        np.savez_compressed(os.path.join(save_path, f'spike_signals{fn_spec}.npz'), spike_signals=spike_signals, t_bins=t_bins, gids=gids)

    return spike_signals, t_bins


def filter_spike_signals(spike_signals, gids, t_bins, sigma, save_path=None, fn_spec=None):

    assert sigma is not None and np.isfinite(sigma) and sigma > 0.0, 'ERROR: Sigma invalid!'
    bin_size = np.min(np.diff(t_bins)).tolist()
    spike_signals = gaussian_filter1d(spike_signals, sigma / bin_size, axis=1)

    if fn_spec is None:
        fn_spec = ''
    if len(fn_spec) > 0:
        fn_spec = f'_{fn_spec}'

    if save_path is not None:
        np.savez_compressed(os.path.join(save_path, f'spike_signals{fn_spec}.npz'), spike_signals=spike_signals, t_bins=t_bins, gids=gids, sigma=sigma)

    return spike_signals


def run_rate_extraction(working_dir, spike_file_names, pool_size=10):
    """ Run extration of EXC firing rates. [PARALLEL IMPLEMENTATION] """

    spike_files = [os.path.join(working_dir, fn) for fn in spike_file_names]

    neuron_info = pd.read_pickle(os.path.join(working_dir, 'neuron_info.pickle'))
    gids = neuron_info[neuron_info['synapse_class'] == 'EXC'].index # Excitatory GIDs extracted from neuron info table

    # Run firing rate extraction [PARALLEL IMPLEMENTATION]
    fct_args = [(f, gids) for f in spike_files]
    t0 = time.time()
    with multiprocessing.Pool(pool_size) as pool:
        pool.map(rate_proc_fct, fct_args)
    print(f'Finished firing rate extraction in {time.time() - t0:.3f}s')


def rate_proc_fct(fct_args):
    """ Firing rate extraction function for parallel implementation. """
    spike_file, gids = fct_args

    fidx = int(os.path.splitext(spike_file)[0].split('_')[-1])
    spikes, _, _ = load_spike_trains(spike_file)
    firing_rates = extract_firing_rates(spikes, gids, save_path=os.path.split(spike_file)[0], fn_spec=f'exc_{fidx}__tmp__')


def extract_firing_rates(spikes, gids, save_path=None, fn_spec=None):
    """ Computes firing rates based on mean inverse inter-spike intervals. """

    isi = [np.diff(np.sort(spikes[spikes[:, 1] == gid, 0])).tolist() for gid in gids] # Inter-spike intervals in ms
    firing_rates = np.array([1e3 / np.mean(_isi) for _isi in isi])

    if fn_spec is None:
        fn_spec = ''
    if len(fn_spec) > 0:
        fn_spec = f'_{fn_spec}'

    if save_path is not None:
        np.savez(os.path.join(save_path, f'firing_rates{fn_spec}.npz'), firing_rates=firing_rates, gids=gids)

    return firing_rates


def merge_rates_to_h5_data_store(working_dir, rate_file_names, data_store_name, do_overwrite=False):
    """ Merges EXC firing rates into existing .h5 store as single dataset for all simulations. """

    rate_files = [os.path.join(working_dir, fn) for fn in rate_file_names]

    # Check that all files exist
    for rate_file in rate_files:
        assert os.path.exists(rate_file), f'ERROR: Rate file "{rate_file}" missing!'

    # Check that .h5 store already exists
    h5_file = os.path.join(working_dir, data_store_name + '.h5')
    assert os.path.exists(h5_file), 'ERROR: h5 store does not exists!'

    # Create backup before writing to data store
    time_stamp = np.round(time.time()).astype(int).astype(str)
    shutil.copy(h5_file, os.path.join(working_dir, f'{data_store_name}__BAK_{time_stamp}.h5'))

    # Check if dataset already exists
    with h5py.File(h5_file, 'r+') as h5f:
        if do_overwrite:
            if 'firing_rates' in h5f:
                print(f'INFO: Firing rates dataset already exists - OVERWRITING!')
                del h5f['firing_rates']
        else:
            assert not 'firing_rates' in h5f, 'ERROR: Firing rates dataset already exists! Please use "do_overwrite=True" to overwrite.'

    # Merge rates of all simulations
    gids = None
    firing_rates = []
    for rate_file in tqdm.tqdm(rate_files):
        fidx = int(os.path.splitext(rate_file)[0].replace('__tmp__', '').split('_')[-1])
        r_dict = np.load(rate_file)
        if gids is None:
            gids = r_dict['gids']
        else:
            assert np.array_equal(gids, r_dict['gids']), 'ERROR: GIDs mismatch!'
        r = r_dict['firing_rates']
        assert len(r) == len(gids), 'ERROR: Rate shape mismatch!'
        firing_rates.append(r)
    firing_rates = np.array(firing_rates)

    # Add rates to .h5 store
    h5f = h5py.File(h5_file, 'r+')
    h5f.create_dataset('firing_rates', data=firing_rates)
    h5f.close()

    print(f'INFO: {len(rate_files)} files merged and added to "{h5_file}"')

    return h5_file


def merge_removed_conns_to_h5_data_store(h5_file, campaign_path, conns_list_fn, do_overwrite=False):
    """ Merges removed connections lists into existing .h5 store as separate datasets for simulations. """

    # Load campaign config
    with open(os.path.join(campaign_path, 'config.json'), 'r') as f:
        config_dict = json.load(f)
    sim_paths = [os.path.join(config_dict['attrs']['path_prefix'], p) for p in config_dict['data']]

    # Extract lists of removed connections
    conns_name = os.path.splitext(conns_list_fn)[0]
    conns_lists = []
    for sidx, spath in enumerate(sim_paths):
        assert os.path.exists(os.path.join(spath, conns_list_fn))
        conns_lists.append(np.load(os.path.join(spath, conns_list_fn)))

    # Check that .h5 store already exists
    assert os.path.exists(h5_file), 'ERROR: h5 store does not exists!'

    # Create backup before writing to data store
    time_stamp = np.round(time.time()).astype(int).astype(str)
    shutil.copy(h5_file, os.path.splitext(h5_file)[0] + f'__BAK_{time_stamp}' + os.path.splitext(h5_file)[-1])

    # Check if dataset already exists
    with h5py.File(h5_file, 'r+') as h5f:
        if do_overwrite:
            if conns_name in h5f:
                print(f'INFO: "{conns_name}" group/dataset already exists - OVERWRITING!')
                del h5f[conns_name]
        else:
            assert not conns_name in h5f, f'ERROR: "{conns_name}" group/dataset already exists! Please use "do_overwrite=True" to overwrite.'

    # Add connection lists to .h5 store
    h5f = h5py.File(h5_file, 'r+')
    grp = h5f.create_group(conns_name)
    for sidx, conns in enumerate(conns_lists):
        grp.create_dataset(f'sim_{sidx}', data=conns)
    h5f.close()

    print(f'INFO: {len(conns_lists)} removed connections lists merged and added to "{h5_file}"')
