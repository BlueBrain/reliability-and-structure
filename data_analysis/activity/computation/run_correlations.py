# SPDX-FileCopyrightText: 2024 Blue Brain Project / EPFL
#
# SPDX-License-Identifier: AGPL-3.0-or-later

'''
Compute correlations of (sub)population
Last updated: 12.2023
'''
import conntility
from scipy.spatial import distance
import numpy as np
import json
import sys 
sys.path.append('../../../library')
from preprocess import load_spike_trains,extract_binned_spike_signals, filter_spike_signals
from dimensionality import correlation

def main():
    with open(sys.argv[1], "r") as fid:
        cfg = json.load(fid)
    print("Running")
    # Get gids 
    conn=conntility.ConnectivityMatrix.from_h5(cfg['connectome']['path'])
    if 'filter' in cfg['connectome']:
        for ftype in cfg['connectome']['filter'].keys():
            conn=conn.index(ftype).isin(cfg['connectome']['filter'][ftype])
    gids=conn.vertices["index"].to_numpy()
    del(conn)

    # Load spike trains and transform to arrays 
    spikes, _, _ = load_spike_trains(f'{cfg["sim_root"]}/{cfg["spikes"]["fname"]}')
    t_max=cfg['spikes']['t_max']
    spike_signals, t_bins = extract_binned_spike_signals(spikes, gids, t_max, bin_size=1.0)
    del(spikes)
    print("Signals loaded")
    # Convolve with gaussian if required
    if 'filter' in cfg['spikes']:
        sigma=cfg['spikes']['filter']['sigma']
        spike_signals = filter_spike_signals(spike_signals, gids, t_bins, sigma)
        print("Signals filtered")

    # Compute correlation and save
    #spike_signals=spike_signals[:100,:] 
    corr=correlation(spike_signals)
    save_path=f'{cfg["sim_root"]}{cfg["save_path"]}'
    np.savez(save_path, corr=corr, gids=gids)
    print(f"Done with {save_path}")
if __name__ == '__main__':
    main()
