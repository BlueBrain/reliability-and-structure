#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 350G
#SBATCH --time=6:00:00
#SBATCH --exclusive 
#SBATCH --job-name=correlations
#SBATCH --account=proj9
#SBATCH --partition=prod
#SBATCH --output=logs/dimension_computation

source /gpfs/bbp.cscs.ch/home/egassant/connalysis/bin/activate 

declare -a 
configs=("activity_dim_from_signals_all_EXC.json"
"activity_dim_from_spikes_all_EXC.json")

for config in ${configs[@]}
do 
echo $config
echo `date`
python -u compute_activity_dimension.py $config
echo `date`
done
echo "Exit code $?"
