#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 350G
#SBATCH --time=6:00:00
#SBATCH --exclusive 
#SBATCH --job-name=correlations
#SBATCH --account=proj9
#SBATCH --partition=prod
#SBATCH --output=logs/add_rc

source /gpfs/bbp.cscs.ch/home/egassant/connalysis/bin/activate 


echo `date`
python -u add_rc_manipulations.py 
echo `date`
echo "Exit code $?"
