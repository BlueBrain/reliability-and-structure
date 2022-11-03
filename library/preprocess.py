'''Description: Functions to pre-process spike trains via gaussian kernel and mean centering
   Author: C. Pokorny
   Date: 11/2022
'''

import h5py
import multiprocessing
import numpy as np
import os
import pandas as pd
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

    # Run preprocessing
    # [SINGLE-THREAD IMPLEMENTATION]
    # for idx in tqdm.tqdm(range(len(spike_files))):
    #     spike_file = spike_files[idx]
    #     spikes, _, _ = load_spike_trains(spike_file)
    #     spike_signals, t_bins = extract_binned_spike_signals(spikes, gids, t_max, bin_size=1.0, save_path=None, fn_spec=None)
    #     spike_signals = filter_spike_signals(spike_signals, gids, t_bins, sigma, save_path=None, fn_spec=None)
    #     spike_signals = mean_center_spike_signals(spike_signals, gids, t_bins, save_path=os.path.split(spike_file)[0], fn_spec=f'exc_{idx}__tmp__')

    # [PARALLEL IMPLEMENTATION]
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
    spike_signals = mean_center_spike_signals(spike_signals, gids, t_bins, save_path=os.path.split(spike_file)[0], fn_spec=f'exc_{fidx}__tmp__')


def merge_into_h5_data_store(working_dir, processed_file_names, data_store_name, split_by_gid=False):
    """ Merge processed EXC spike signals into .h5 store as separate data sets sims and optionally, GIDs. [WITH COMPRESSION] """

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

    t_bins = np.arange(0.0, t_max + bin_size, bin_size)
    spike_signals = np.array([np.histogram(spikes[spikes[:, 1] == gid, 0], bins=t_bins)[0] for gid in gids]).astype(float)

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


def mean_center_spike_signals(spike_signals, gids, t_bins, save_path=None, fn_spec=None):

    spike_signals = spike_signals - np.mean(spike_signals, 1, keepdims=True)

    if fn_spec is None:
        fn_spec = ''
    if len(fn_spec) > 0:
        fn_spec = f'_{fn_spec}'

    if save_path is not None:
        np.savez_compressed(os.path.join(save_path, f'spike_signals{fn_spec}.npz'), spike_signals=spike_signals, t_bins=t_bins, gids=gids)

    return spike_signals

