#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 350G
#SBATCH --time=20:00:00
#SBATCH --job-name=nbds
#SBATCH --account=proj9
#SBATCH --partition=prod
#SBATCH --output=logs/nbds_connectomes

source /gpfs/bbp.cscs.ch/home/egassant/connalysis/bin/activate 

declare -a 
configs=("Celegans_nbds.json"
"Drosophila_nbds.json"
"MICrONS_nbds.json"
"BBP_nbds.json")

for config in ${configs[@]}
do 
echo $config
python -u run_nbd_basics.py $config
done
echo "Exit code $?"
