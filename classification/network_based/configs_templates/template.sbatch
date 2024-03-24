#!/bin/bash
#SBATCH -N 1 # number of nodes
#SBATCH -n 8 # number of cores
#SBATCH --mem 256G # memory pool for all cores
#SBATCH -o TriDy-tools/out-err/#SSHORT-#FSHORT-#JOBNUM.out # STDOUT
#SBATCH -e TriDy-tools/out-err/#SSHORT-#FSHORT-#JOBNUM.err # STDERR
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=1
#SBATCH --job-name=#SSHORT-#FSHORT-#JOBNUM

python 'TriDy-tools/runfiles/pipeline-#SSHORT.py' 'TriDy-tools/configs/#SSHORT-#FSHORT/#JOBNUM.json'
