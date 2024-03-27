
Here we list the scripts provided and describe what properties they are used to compute

####  1. Simplex counts and % of reciprocal connections on subgraph of n-simplices, for all connectomes and their controls. 

- Base script:  run_basic_properties.py
- Config files: <connectome>_basic.json
- Run script: run_all_basic.sh  

 ####  2. Reciprocal connections for all connectomes and their controls. 

- Base script: run_rc_across_connectomes.py
- Config files: rc_den_<connectome>.json
- Run script:  run_all_rc_densities.sh 


 ####  3. Neighborhood properties for all connectomes and their controls 

- Base script: run_nbd_basics.py
- Config files: <connectome>_nbds_basic.json
- Run script:  run_all_nbds.sh



## TODO

- [ ] Neighborhood complexity
  - [ ] run_max_simplex_lists.py
  - [ ] run_simplex_lists.py
- [ ] Manipulations
  - [ ] Manipulation generation
  - [ ] Computing simplex counts (simplex_counts_manipulated_connectomes.py (push me), separate for enhanced) 




