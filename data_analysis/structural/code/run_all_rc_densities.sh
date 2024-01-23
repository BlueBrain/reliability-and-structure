#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 350G
#SBATCH --time=5:00:00
#SBATCH --job-name=rc_den
#SBATCH --account=proj9
#SBATCH --partition=prod
#SBATCH --output=logs/rc_den

source /gpfs/bbp.cscs.ch/home/egassant/connalysis/bin/activate 

declare -a 
configs=(#"rc_den_Celegans.json"
#"rc_den_Drosophila.json"
#"rc_den_MICrONS.json"
"rc_den_BBP.json")

for config in ${configs[@]}
do 
echo $config
python -u run_rc_across_connectomes.py $config
done
echo "Exit code $?"
