#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 350G
#SBATCH --time=6:00:00
#SBATCH --job-name=rel_bootstrap
#SBATCH --account=proj9
#SBATCH --partition=prod

source /gpfs/bbp.cscs.ch/home/egassant/connalysis/bin/activate 
python -u compute_reliability_bootstrap.py $1