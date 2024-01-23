'''
Compute rc densities for original and many samples of a control
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
        rc_original_and_controls(cfg)

if __name__ == '__main__':
    main()


