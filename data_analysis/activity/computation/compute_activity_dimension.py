# SPDX-FileCopyrightText: 2024 Blue Brain Project / EPFL
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import json 
import numpy as np
import pandas as pd
import conntility
import sys 
sys.path.append('../../../library')
from preprocess import load_spike_trains, extract_binned_spike_signals
from dimensionality import get_dimensions_nbds

def main():
    with open(sys.argv[1], "r") as fid:
        cfg = json.load(fid)
    ### Load data
    # Load correlations
    fname=f"{cfg['sim_root']}{cfg['correlations_fname']}"
    corr=np.load(fname)["corr"]
    gids=np.load(fname)["gids"]
    # Load connectivity matrix
    fname=f"{cfg['sim_root']}{cfg['connectome']['path']}"
    conn=conntility.ConnectivityMatrix.from_h5(fname)
    if 'filter' in cfg['connectome']:
        for ftype in cfg['connectome']['filter'].keys():
            conn=conn.index(ftype).isin(cfg['connectome']['filter'][ftype])
    # Check consistency between activity and structure data
    assert np.array_equal(conn.gids, gids), "Gids of connectome dont'match with gids of correlations"
    # Load spike trains to resolve nans
    spikes, _, _ = load_spike_trains(f'{cfg["sim_root"]}/{cfg["spikes"]["fname"]}')
    t_max=cfg['spikes']['t_max']
    spike_signals, t_bins = extract_binned_spike_signals(spikes, gids, t_max, bin_size=1.0)
    
    ### Get centers for which the computation is carried out
    if cfg['centers']== 'all':
        all_nodes=True
        centers=None
    else:
        all_nodes=False
        centers=np.load(cfg['centers'])
    
    ### Compute dimensions and save
    df=get_dimensions_nbds(conn, corr, spike_signals, all_nodes, centers, not_firing_corr=False)
    fname=f"{cfg['sim_root']}{cfg['save_path']}"
    df.to_pickle(fname)
    print(f"Done with {cfg['sim_root']}")
    
if __name__ == '__main__':
    main()
