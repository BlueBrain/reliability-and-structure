It is recommended to create a new virtual environment within which to run the pipeline, by doing:

```
python -m venv topsamp
source ./topsamp/bin/activate
```

To install run:
```
sh install.sh
```

Then run with:
```
python run.py
```

The scripts included here are:
- run.py : This runs the pipeline which does the PCA classification
- create_results_df.py : This collates the results from the pipeline into a single database, it is called automatically by run.py


# TODO:
- [X] community_database.pkl is already in the zenodo.  Can this code be structured as the user gives a path to the zenodo files to run this? DONE: but needs testing once the zenodo is ready
- [X] Add quick description of create_results_df.py
- [X] Add quick description of run.py
 
