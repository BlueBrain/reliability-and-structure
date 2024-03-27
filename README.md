# Reliability and structure (SHOULD WE CHANGE THIS NAME?)


Study of network structure and how it shapes the robustness - reliability - efficiency struggle in biological neural networks as described in this publication. 

[![DOI:10.1101/2024.03.15.585196](http://img.shields.io/badge/DOI-10.1101/2024.03.15.585196-B31B1B.svg)](https://doi.org/10.1101/2024.03.15.585196)

The repository is structured as follows:
 
- **library:** Library of functions for all the analysis related to the publication except for classficiation
- **data_analysis:** In this directory we provide all the scripts use to compute the different network metrics and their relation to function.
  - **structural:** Subdirectory where the analysis of purely structural properties for all connectomes and their corresponding controls is performed.
      - **code:** Scripts that generate the data.  [README](https://github.com/danielaegassan/reliability_and_structure/blob/main/data_analysis/structural/code/README.md)
      - **visualization_and_notebooks:** Scripts or notebooks to visualize data or generate figures 
  - **activity:** Subdirectory where the analysis of properties that relate to function or link function to structure in BBP and MICrONS is performed.
      - **computation:** Scripts that generate the data. [README](https://github.com/danielaegassan/reliability_and_structure/blob/main/data_analysis/activity/computation/README.md)
      - **visualization:** Scripts or notebooks to visualize data or generate figures
- **classification:**
  - **PCA_method:**
  - **network_based:**
  - **visualization:**

Local README files provide a description of the scripts used for computation.
 

## Citation  
If you use this software, kindly use the following BibTeX entry for citation:

```
@article{egas2024efficiency,
  title={Efficiency and reliability in biological neural network architectures},
  author={Egas Santander, Daniela and Pokorny, Christoph and Ecker, Andr{\'a}s and Lazovskis, J{\=a}nis and Santoro, Matteo and Smith, Jason P and Hess, Kathryn and Levi, Ran and Reimann, Michael W},
  journal={bioRxiv},
  pages={2024--03},
  year={2024},
  publisher={Cold Spring Harbor Laboratory},
  doi = {10.1101/2024.03.15.585196}
}
```

# TODO:
- [ ] Rename directories to consistent names.
- [ ] Fill in required descriptions above
- [ ] Add copy right and link to the paper
- [ ] Add license
- [ ] Add requirements.txt
- [ ] Installable library?
- [ ] Link to zenodo and say something about dowloading this somewhere? Add zenodo DOI?
- [ ] Name?
- [ ] Delete unnecessary files
