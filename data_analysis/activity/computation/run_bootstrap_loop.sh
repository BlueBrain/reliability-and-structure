#!/bin/bash

declare -a 
sims=("BlobStimReliability_O1v5-SONATA_RecipRemoval_StructDim56"
"BlobStimReliability_O1v5-SONATA_RecipRemoval_StructDim56_456"
"BlobStimReliability_O1v5-SONATA_RecipRemoval_StructDim456"
"BlobStimReliability_O1v5-SONATA_RecipRemoval_Unstruct-0"
"BlobStimReliability_O1v5-SONATA_RecipRemoval_Unstruct-3"
"BlobStimReliability_O1v5-SONATA_RecipRemoval_Unstruct-1"
"BlobStimReliability_O1v5-SONATA_RecipRemoval_Unstruct-2"
"BlobStimReliability_O1v5-SONATA_Baseline"
"BlobStimReliability_O1v5-SONATA_ConnRewired_mc2EE_Order1"
"BlobStimReliability_O1v5-SONATA_ConnRewired_mc2EE_Order2"
"BlobStimReliability_O1v5-SONATA_ConnRewired_mc2EE_Order3"
"BlobStimReliability_O1v5-SONATA_ConnRewired_mc2EE_Order4"
"BlobStimReliability_O1v5-SONATA_ConnRewired_mc2EE_Order5")

for sim in ${sims[@]}
do 
echo $sim
sbatch --output=logs/slurm_${sim} run_bootstrap_single_sim.sh $sim
done

echo "Exit code $?"


