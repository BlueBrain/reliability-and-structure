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
from coupling_coefficient import coupling_coefficient, cc_ctrls

STRUCTURAL_DATA_DIR = "/gpfs/bbp.cscs.ch/home/egassant/reliability_and_structure/data_analysis/data"
PROJ_DIR = "/gpfs/bbp.cscs.ch/data/scratch/proj9/bisimplices/bbp_workflow/7b381e96-91ac-4ddd-887b-1f563872bd1c/"


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
    conn_mat = load_connectome(STRUCTURAL_DATA_DIR, "BBP")
    spikes, gids, t_max = load_spike_trains(os.path.join(PROJ_DIR, "toposample_input", "raw_spikes_exc.npy"))
    t_max = np.ceil(t_max) + 1  # not sure if this is needed...
    ccs = coupling_coefficient(spikes, gids, t_max)
    cc_ctrls = cc_ctrls(spikes, gids, t_max)
    ccs, cc_ctrls = reindex_ccs(ccs, cc_ctrls, gids, conn_mat)
    df = pd.concat([ccs, cc_ctrls], axis=1)
    df.to_pickle(os.path.join(PROJ_DIR, "working_dir", "coupling_coefficients.pkl"))