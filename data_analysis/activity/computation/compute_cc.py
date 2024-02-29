"""
Computes coupling coefficient (correlation of (binned) spike train with the mean of the whole population)
author: András Ecker, last update: 02.2024
"""

import os
import numpy as np
import pandas as pd
import sys
sys.path.append("../../../library")
from structural_basic import *
from preprocess import load_spike_trains
from coupling_coefficient import *

STRUCTURAL_DATA_DIR = "/gpfs/bbp.cscs.ch/home/egassant/reliability_and_structure/data_analysis/data"
FUNCTIONAL_DATA_DIR = "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/assemblyfire/MICrONS"
PROJ_DIR = "/gpfs/bbp.cscs.ch/data/scratch/proj9/bisimplices/bbp_workflow/7b381e96-91ac-4ddd-887b-1f563872bd1c/"


def get_microns_data(npzf_name, conn_mat):
    """Loads ''spikes'' and oracle score from MICrONS functional data (saved to npz in `assemblyfire`),
    maps idx to the structural connectome, calculates coupling coefficients
    and padds everything with nans to have the same shape (and index) as the structural data"""
    tmp = np.load(npzf_name)
    idx, spikes = tmp["idx"], tmp["spikes"]
    # drop duplicated idx and get oracle scores
    unique_idx, counts = np.unique(idx, return_counts=True)
    idx_tmp = np.in1d(idx, unique_idx[counts == 1])
    idx, spikes = idx[idx_tmp], spikes[idx_tmp, :]
    oracle_scores = pd.Series(tmp["oracle_scores"][idx_tmp], index=idx)
    oracle_scores = oracle_scores.loc[oracle_scores.notna()]
    # match idx to structural data
    valid_idx = conn_mat.vertices.id[np.isin(conn_mat.vertices.id, oracle_scores.index)]
    oracle_scores = oracle_scores.loc[oracle_scores.index.isin(valid_idx)]
    valid_idx_tmp = pd.Series(valid_idx.index.to_numpy(), index=valid_idx.to_numpy())
    oracle_scores = pd.Series(oracle_scores.to_numpy(), index=valid_idx_tmp.loc[oracle_scores.index].to_numpy()).sort_index()
    # index out spikes as well (and sort corresponding gids)
    idx_tmp = np.where(np.in1d(idx, valid_idx_tmp.index.to_numpy()))[0]
    gids = valid_idx_tmp.loc[idx[idx_tmp]].to_numpy()
    sort_idx = np.argsort(gids)
    spikes, gids = spikes[idx_tmp[sort_idx], :], gids[sort_idx]
    # get rates and coupling coefficients
    rates = pd.Series(np.mean(spikes, axis=1), index=gids)
    ccs = pd.Series(coupling_coefficient_loo(spikes), index=gids)
    data = cc_loo_ctrls(spikes)
    columns = ["coupling_coeff_ctrl_%i" % i for i in range(data.shape[1])]
    cc_ctrls = pd.DataFrame(data, index=gids, columns=columns)
    # padd everything with NaNs
    idx = conn_mat.vertices.index.to_numpy()
    data = np.full(len(idx), np.nan)
    data[rates.index.to_numpy()] = rates.to_numpy()
    rates = pd.Series(data, index=idx, name="rate")
    data = np.full(len(idx), np.nan)
    data[oracle_scores.index.to_numpy()] = oracle_scores.to_numpy()
    oracle_scores = pd.Series(data, index=idx, name="oracle_score")
    data = np.full(len(idx), np.nan)
    data[ccs.index.to_numpy()] = ccs.to_numpy()
    ccs = pd.Series(data, index=idx, name="coupling_coeff")
    data = np.full((len(idx), len(columns)), np.nan)
    data[cc_ctrls.index.to_numpy()] = cc_ctrls.to_numpy()
    cc_ctrls = pd.DataFrame(data, index=idx, columns=columns)
    return rates, oracle_scores, ccs, cc_ctrls


def get_all_microns_data(session_idx, scan_idx, conn_mat):
    """Calls `get_functional_data()` from above for all scans and concatenates results"""
    dfs = []
    for session_id, scan_id in zip(session_idx, scan_idx):
        name_tag = "session%i_scan%i" % (session_id, scan_id)
        rates, oss, ccs, cc_ctrls = get_functional_data(os.path.join(FUNCTIONAL_DATA_DIR,
                                                                     "MICrONS_%s.npz" % name_tag), conn_mat)
        df = pd.concat([rates, oss, ccs, cc_ctrls], axis=1)
        df.columns = pd.MultiIndex.from_arrays([np.full(len(df.columns), name_tag), df.columns.to_numpy()])
        dfs.append(df)
    return pd.concat(dfs, axis=1)


def reindex_ccs(ccs, cc_ctrls, gids, conn_mat):
    """Returns cc values (and their controls) with the same indices as `conn_mat`"""
    idx = pd.Series(conn_mat.vertices.index.to_numpy(), index=conn_mat.vertices["index"].to_numpy())
    data = np.full(len(idx), np.nan)
    data[idx.loc[gids]] = ccs
    ccs = pd.Series(data, index=idx.to_numpy(), name="coupling_coeff")
    columns = ["coupling_coeff_ctrl_%i" % i for i in range(cc_ctrls.shape[1])]
    data = np.full((len(idx), len(columns)), np.nan)
    data[idx.loc[gids]] = cc_ctrls
    cc_ctrls = pd.DataFrame(data, index=idx.to_numpy(), columns=columns)
    return ccs, cc_ctrls


if __name__ == "__main__":
    session_idx = [4, 5, 6, 6, 6, 7, 8, 9]
    scan_idx = [7, 7, 2, 4, 7, 4, 5, 3]
    conn_mat = load_connectome(STRUCTURAL_DATA_DIR, "MICrONS")
    df = get_all_microns_data(session_idx, scan_idx, conn_mat)
    df.to_pickle(os.path.join(FUNCTIONAL_DATA_DIR, "MICrONS_functional_summary.pkl"))

    conn_mat = load_connectome(STRUCTURAL_DATA_DIR, "BBP")
    spikes, gids, t_max = load_spike_trains(os.path.join(PROJ_DIR, "toposample_input", "raw_spikes_exc.npy"))
    t_max = np.ceil(t_max) + 1  # not sure if this is needed...
    ccs = coupling_coefficient(spikes, gids, t_max)
    cc_ctrls = cc_ctrls(spikes, gids, t_max)
    ccs, cc_ctrls = reindex_ccs(ccs, cc_ctrls, gids, conn_mat)
    df = pd.concat([ccs, cc_ctrls], axis=1)
    df.to_pickle(os.path.join(PROJ_DIR, "working_dir", "coupling_coefficients.pkl"))