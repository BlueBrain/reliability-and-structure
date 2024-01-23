#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 350G
#SBATCH --time=4:00:00
#SBATCH --job-name=plastic_rewire
#SBATCH --account=proj9
#SBATCH --partition=prod
#SBATCH --output=logs/plasticity_rewiring_100s

source /gpfs/bbp.cscs.ch/home/egassant/connalysis/bin/activate 

declare -a 
configs=("V5_placement_4k_rewire.json"
"V5_placement_100k.json"
"V5_placement_200k.json"
"V5_placement_300k.json"
"V5_placement_400k.json"
"V5_placement_500k.json"
"V5_placement_20k_rewire.json"
"V5_placement_66k_rewire.json"
"V5_placement_270k.json"
"V5_placement_470k.json"
"V5_placement_670k.json")

for config in ${configs[@]}
do 
echo $config
python -u run_plasticity_rewiring.py $config
done
echo "Exit code $?"