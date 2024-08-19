#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 350G
#SBATCH --time=20:00:00
#SBATCH --job-name=basic_props_network
#SBATCH --account=proj9
#SBATCH --partition=prod
#SBATCH --output=logs/basic_props_network

source /gpfs/bbp.cscs.ch/home/egassant/connalysis/bin/activate 

declare -a 
configs=(#"Celegans_basics.json"
#"Drosophila_basics.json"
#"MICrONS_basics.json"
#"BBP_basics.json"
"MICrONS_EM_error_precision_basics.json"
"MICrONS_EM_error_recall_basics.json"
"MICrONS_EM_error_partner_basics.json")

for config in ${configs[@]}
do 
echo $config
python -u run_basic_properties.py $config
done
echo "Exit code $?"