# SPDX-FileCopyrightText: 2024 Blue Brain Project / EPFL
#
# SPDX-License-Identifier: AGPL-3.0-or-later

'''
Compute basic topological network properites of the neighborhoods of a connectome and its controls
Author(s): Daniela Egas S.
Last updated: 12.2023
'''
import json 
import sys
sys.path.append('../../../library')
from structural_basic import *

def main():
    with open(sys.argv[1], "r") as f:
        cfg=json.load(f)
        print(f'Computing {cfg["connectome"]["name"]}')
        compute_basics_over_nbds_ori_and_controls(cfg)
        print(f'Done with {cfg["connectome"]["name"]}')

if __name__ == '__main__':
    main()
    