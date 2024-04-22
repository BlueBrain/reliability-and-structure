# SPDX-FileCopyrightText: 2024 Blue Brain Project / EPFL
#
# SPDX-License-Identifier: AGPL-3.0-or-later

''' 
Compute a rewiring of the V5 circuit according to input config
Author(s): Daniela Egas S. 
Last updated 12.2023
'''

import json
import time
import sys
import numpy

sys.path.append('/gpfs/bbp.cscs.ch/home/egassant/excitation_inhibition_topology')
from helpers import network


def main():
    with open(sys.argv[1], "r") as fid:
        cfg = json.load(fid)
    numpy.random.seed(0)
    M = network.load_network(cfg["connectome"]["loading"])
    start=time.time()
    network.override_connectivity(M, cfg["connectome"]["override"])    
    M.to_h5(cfg["connectome"]["save"])
    print(f"Time to compute override {(time.time()-start)/60:.2f} min")


if __name__ == "__main__":
    main()