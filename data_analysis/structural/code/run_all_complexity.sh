#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 350G
#SBATCH --time=12:00:00
#SBATCH --job-name=complexity_network
#SBATCH --account=proj9
#SBATCH --partition=prod
#SBATCH --output=logs/complexity_analysis

source /gpfs/bbp.cscs.ch/home/egassant/connalysis/bin/activate 

declare -a 
run_files=("complexity_Drosophila.py"
"complexity_Celegans.py"
"complexity_MICrONS.py"
"complexity_BBP.py")

for file in ${run_files[@]}
do 
echo $file
python -u $file
done
echo "Exit code $?"