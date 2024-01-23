#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 350G
#SBATCH --time=12:00:00
#SBATCH --exclusive 
#SBATCH --job-name=correlations
#SBATCH --account=proj9
#SBATCH --partition=prod
#SBATCH --output=logs/correlation_signals

source /gpfs/bbp.cscs.ch/home/egassant/connalysis/bin/activate 

declare -a 
configs=("corr_signals_exc.json")
#"corr_spikes_exc.json"
#"corr_signals_exc.json"

for config in ${configs[@]}
do 
echo $config
python -u run_correlations.py $config
done
echo "Exit code $?"
