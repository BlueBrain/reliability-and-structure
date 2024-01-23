#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 350G
#SBATCH --time=5:00:00
#SBATCH --job-name=sc_enhanced
#SBATCH --account=proj102
#SBATCH --partition=prod
#SBATCH --output=logs/reliab_all

source /gpfs/bbp.cscs.ch/home/egassant/connalysis/bin/activate 
python -u compute_reliab_basic.py