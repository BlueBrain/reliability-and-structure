Here we list the scripts used to compute activity related properties i.e., those involving functional data.  Only available for the MICrONS and BBP connectomes.

### Reliability and efficiency simulations (short sims, multiple seeds) or MICrONS data:
####  1. Extract spike trains from BBP simulations
  - Jupyter notebook: [run_spike_extraction.ipynb](./run_spike_extraction.ipynb)

####  2. Preprocess spike trains from BBP simulations
  - Jupyter notebook: [run_spike_preprocessing.ipynb](./run_spike_preprocessing.ipynb)
  
####  3. Compute reliabilities of BBP connectomes 
  - Base script:  [compute_reliab_basic.py](./compute_reliab_basic.py)
  - Run script: [run_reliab_basic.sh](./run_reliab_basic.sh)

####  4. Compute coupling coefficient 
  - Base script:  [compute_cc.py](./compute_cc.py)

####  5. Compute efficiency
  - Base script:  [compute_activity_dimension.py](./compute_activity_dimension.py)
  - Run script: [run_dimensions_activity.sh](./run_dimensions_activity.sh)

### Toposample simulations (long sim, single seed):
####  1. Extract input data for running TriDy/TopoSampling [classification](../../../classification) pipeline
  - Jupyter notebook: [toposample_preparation.ipynb](./toposample_preparation.ipynb)
