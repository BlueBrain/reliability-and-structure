### PCA featurization 

Here we describe how to run the classification pipeline with PCA featurization only.  The selection process of the neighborhoods used for classification is explained in more detail in the network featurization section 
[here.](https://github.com/danielaegassan/reliability_and_structure/blob/main/classification/network_based/README.md)

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
