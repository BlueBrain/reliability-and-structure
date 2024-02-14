"""
Functions that calculate coupling coefficent
(Pearson correlation of binned spikes/calcium traces with the mean centered population average)
authors: András Ecker, Daniela Egas Santander, Michael W. Reimann; last update: 02.2024
"""

import numpy as np
from preprocess import extract_binned_spike_signals


def coupling_coefficient(spikes, gids, t_max, bin_size=5):
    """Fast implementation of coupling coefficient: (corr of mean centered binned spikes
    with the avg. of the binned spikes all neurons) by Michael"""
    binned_spikes, _ = extract_binned_spike_signals(spikes, gids, t_max, bin_size=bin_size)
    binned_spikes = binned_spikes - binned_spikes.mean(axis=1, keepdims=True)
    mn_pop = binned_spikes.mean(axis=0)
    A = np.dot(binned_spikes, mn_pop.reshape((-1, 1)))[:, 0]
    B = np.sqrt(np.var(mn_pop) * np.var(binned_spikes, axis=1)) * binned_spikes.shape[1]
    return A / B


def _shuffle_spikes(spikes, seed):
    """Shuffle spiking gids (thus keeps single cell firing rates the same)"""
    np.random.seed(seed)
    idx = np.random.permutation(spikes.shape[0])
    return np.vstack([spikes[:, 0], spikes[idx, 1]]).T


def cc_ctrls(spikes, gids, t_max, bin_size=5, n_ctrls=10, seed=12345):
    """Generates surrogate datasets by shuffling spike times and gets their CC values"""
    cc_ctrls = np.zeros((len(np.unique(spikes[:, 1])), n_ctrls), dtype=np.float32)
    for i in range(n_ctrls):
        shuffled_spikes = _shuffle_spikes(spikes, seed=seed + i)
        cc_ctrls[:, i] = coupling_coefficient(shuffled_spikes, gids, t_max, bin_size=bin_size)
    return cc_ctrls


def coupling_coefficient_loo(traces):
    """As `coupling_coefficient()` above, but instead of using the average of all mean centered binned spikes
    it's leave-one-out (`_loo`), where the one left out from the average is the one we correlate it with.
    While this is the mathematically precise def. of CC, leaving one out shouldn't matter for 10k+ neurons
    and as the above one is way faster, that's the recommended one for spikes,
    while this one for (deconvolved) calcium traces (hence what's `binned_spikes` above is `traces` here).
    (This one can still be called on spikes for testing purposes, just make sure to bin them outside this function)"""
    ccs = np.zeros(traces.shape[0], dtype=np.float32)
    for i in range(1, traces.shape[0] - 1):
        other_rows = np.concatenate([traces[:i, :], traces[i+1:, :]])
        other_rows = other_rows - other_rows.mean(axis=1, keepdims=True)
        ccs[i] = np.corrcoef(traces[i, :], np.mean(other_rows, axis=0))[0, 1]
    # deal with first and last rows separately
    other_rows = traces[1:, :]
    other_rows = other_rows - other_rows.mean(axis=1, keepdims=True)
    ccs[0] = np.corrcoef(traces[0, :], np.mean(other_rows, axis=0))[0, 1]
    other_rows = traces[:-1, :]
    other_rows = other_rows - other_rows.mean(axis=1, keepdims=True)
    ccs[-1] = np.corrcoef(traces[-1, :], np.mean(other_rows, axis=0))[0, 1]
    return ccs


def _shuffle_along_axis(traces, axis, seed):
    """Independently shuffle values within rows/columns of array"""
    np.random.seed(seed)
    idx = np.random.rand(*traces.shape).argsort(axis=axis)
    return np.take_along_axis(traces, idx, axis=axis)


def cc_loo_ctrls(traces, n_ctrls=10, seed=12345):
    """Generates surrogate datasets by row-wise shuffling traces and gets their leave-one-out CC values"""
    cc_ctrls = np.zeros((traces.shape[0], n_ctrls), dtype=np.float32)
    for i in range(n_ctrls):
        shuffled_traces = _shuffle_along_axis(traces, axis=1, seed=seed + i)
        cc_ctrls[:, i] = coupling_coefficient_loo(shuffled_traces)
    return cc_ctrls

