
Here we list the scripts provided and describe what properties they are used to compute

####  1. Simplex counts and % of reciprocal connections on subgraph of n-simplices, for all connectomes and their controls. 

- Base script:  run_basic_properties.py
- Config files: `connectome`_basic.json
- Run script: run_all_basic.sh  

 ####  2. Reciprocal connections for all connectomes and their controls. 

- Base script: run_rc_across_connectomes.py
- Config files: rc_den_`connectome`.json
- Run script:  run_all_rc_densities.sh


 ####  3. Neighborhood properties for all connectomes and their controls 

- Base script: run_nbd_basics.py
- Config files: `connectome`_nbds_basic.json
- Run script:  run_all_nbds.sh

####  4. Neighborhood complexity for all connectomes and their controls 

- Base scripts: complexity_`connectome`.py
- Run script:  run_all_complexity.sh

####  5. List maximal simplices for all connectomes and their controls 

- Base scripts: run_max_simplex_lists.py
- Run script:  run_all_simplex_lists.sh

####  6. Generate manipulations with reduced complexity 
 - Base script: **add me**
   
####  7. Generate manipulations with removed reciprocal connections 
 - Base script: **add me**

####  8. Generate manipulations with added reciprocal connections 
 - Base script: add_rc_manipulations.py

####  9. Generate manipulations with enhnaced complexity 
 - Base script: run_plasticity_rewiring.py
 - Config files: V5_placement_`N`k.json for N = 100, 200, 300, 400, 500, 670
 - Run script:  run_plasticity_rewiring.sh

####  10. Compute simplex counts of manipulated connectomes
 - All manipulations except enhance complexity
   - Base script: simplex_counts_manipulated_connectomes.py
 - For manipulations with enhanced complexity
   - Base script: simplex_counts_enhanced.py
   - Run script: run_simplex_counts_enhanced.sh



## TODO

- [x] Neighborhood complexity
- [x] run_max_simplex_lists.py
- [ ] run_simplex_lists.py
- [ ] Manipulations
  - [x] Add rc
  - [ ] Remove rc
  - [x] Enhance complexity
  - [ ] Reduce complexity
  - [ ] Computing simplex counts (simplex_counts_manipulated_connectomes.py (push me), separate for enhanced) 




