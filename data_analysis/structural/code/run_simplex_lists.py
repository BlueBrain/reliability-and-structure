# SPDX-FileCopyrightText: 2024 Blue Brain Project / EPFL
#
# SPDX-License-Identifier: AGPL-3.0-or-later

'''
Compute simplex lists for all connectomes and their controls
Author(s): Daniela Egas S.
Last updated: 04.2024
'''

import sys
sys.path.append('../../../library')
from structural_basic import *


def main():
    for conn in ['Celegans', 'Drosophila', 'MICrONS', 'BBP']:
        cfg={
            'connectome': {
                'data_dir': '../../data',
                'name': conn},
            'save_path': '../../data',
            'analyses': {
                'list_simplices_by_dimension': {
                    'save_suffix': '',
                    'kwargs': {
                        'threads': 10,
                        'max_simplices':False,
                    },
                    "controls":{
                        "seeds":[10],
                        "types":{
                            "configuration_model":{},
                            "ER_model":{}
                        }
                    }
                }
            }}
    
        print(f'Computing {cfg["connectome"]["name"]}')
        compute_basic_props(cfg)
        print(f'Done with {cfg["connectome"]["name"]}')

    

if __name__ == '__main__':
    main()
    