#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 350G
#SBATCH --time=20:00:00
#SBATCH --job-name=other_metrics
#SBATCH --account=proj9
#SBATCH --partition=prod
#SBATCH --output=logs/other_metrics

source /gpfs/bbp.cscs.ch/home/egassant/connalysis/bin/activate 

python -u compute_other_metrics.py 
echo "Exit code $?"