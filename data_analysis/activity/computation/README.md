
Here we list the scripts provided and describe what properties they are used to compute **in progress**

### Reliability simulations (short sims, multiple seeds):
####  1. Extract spike trains from BBP simulations
  - Jupyter notebook: [run_spike_extraction.ipynb](./run_spike_extraction.ipynb)

####  2. Preprocess spike trains from BBP simulations
  - Jupyter notebook: [run_spike_preprocessing.ipynb](./run_spike_preprocessing.ipynb)
  
####  3. Compute reliabilities of BBP connectomes 
  - Base script:  [compute_reliab_basic.py](./compute_reliab_basic.py) (push me)
  - Config files: 
  - Run script: 

### Toposample simulations (long sim, single seed):
####  1. Extract input data for running TriDy/TopoSampling [classification](../../classification) pipeline
  - Jupyter notebook: [toposample_preparation.ipynb](./toposample_preparation.ipynb)


## TODO

- [x] Preprocessing
  - [x] BBP 
- [ ] Compute CC
  - [ ] MICrONS
  - [ ] BBP, should Michael's original notebook go away?
- [ ] Compute dimension (efficiency)
  - [ ] BBP
- [ ] Reliability clean above
