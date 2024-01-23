'''
Compute basic topological network properites of a connectome and its controls
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
        compute_basic_props(cfg)
        print(f'Done with {cfg["connectome"]["name"]}')

if __name__ == '__main__':
    main()
    