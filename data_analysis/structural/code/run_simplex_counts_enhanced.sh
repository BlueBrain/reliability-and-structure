#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 350G
#SBATCH --time=22:00:00
#SBATCH --job-name=sc_enhanced
#SBATCH --account=proj102
#SBATCH --partition=prod
#SBATCH --output=logs/sc_enhanced

source /gpfs/bbp.cscs.ch/home/egassant/connalysis/bin/activate 
python -u simplex_counts_enhanced.py