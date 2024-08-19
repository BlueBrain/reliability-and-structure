### PCA featurization 

This is the pipeline used to run the classificaiton experiment with PCA featurization.

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
