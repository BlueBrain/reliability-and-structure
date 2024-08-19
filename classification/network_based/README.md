# Network featurization

This section describes how to run the classiciation pipeline using several network metrics as features, as opposed to PCA of the activity.

We refer to the BlueBrian excitatory-excitatory circuit just as "circuit" and describe how neghbourhoods where selected (in most cases, the number of neighbourhoods selected is 50) in order to then classify stimuli and adddress the accuracy of classification.

## Set up

It is recommended to create a new virtual environment within which to run the pipeline, by doing:

    python -m venv network_based
    source ./network_based/bin/activate

To install the necessary packages and TriDy run the code below (this mirrors the installation instructions for TriDy):

    sh install.sh

To compute each group of parameters, run the code below. Some parameters take much longer than others (in particular spectral and topological parameters):
    
    python compute_parameters.py spectral
    python compute_parameters.py spectral_reverse
    python compute_parameters.py simplices
    python compute_parameters.py degrees
    python compute_parameters.py cc
    python compute_parameters.py dc
    python compute_parameters.py nbc
    python compute_parameters.py rc

The rest of the code is contained within the two Jupyter notebooks, as Python code to be executed in each cell and as code to be executed in a terminal.
1. Execute the cells in `1_choose_centers.ipynb` to make the appropriate selections of neighbourhoods. The article accompanying this repo, the notebook, and farther on in this file contains a description of these methods.
2. Execute the code in the cells in `2_run_pipeline.ipynb` in the terminal to run the classification pipeline.

The final output will be dataframes in the folder `TriDy-tools/dataframes/`, one dataframe for each featurization parameter.

## Description of method

Neighbourhoods (a single vertex, its adjacent vertices, and all edges among them) are selected in several ways. The first step is to compute values (such as spectral gaps and clustering coefficients) for every neighbourhood, in particular our density parameter `rc_per_nodes`, which couts the number of pairs of reciprocal connections in a neighbourhood, divided by the number of nodes in the neighbourhood.

### Random selections

Select randomly:
1. neighbourhoods from the full circuit
2. neighbourhoods from the 1% of neighbourhoods with lowest density paramater value
3. neighbourhoods from the 1% of neighbourhoods with highest density paramater value

### Single selections

For every parameter, select
1. neighbourhoods having the highest paramater value from the full circuit
2. neighbourhoods having the lowest paramater value from the full circuit

### Double selections

Restrict to the 1% of neighbourhoods having the lowest density paramater value. The for every other parameter, select
1. neighbourhoods having the highest paramater value from that subcircuit
2. neighbourhoods having the lowest paramater value from that subcircuit
