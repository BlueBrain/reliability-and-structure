#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 350G
#SBATCH --time=24:00:00
#SBATCH --job-name=simplex_lists
#SBATCH --account=proj9
#SBATCH --partition=prod
#SBATCH --output=logs/simplex_lists_max

source /gpfs/bbp.cscs.ch/home/egassant/connalysis/bin/activate 

#python -u run_simplex_lists.py 
python -u run_max_simplex_lists.py 


echo "Exit code $?"