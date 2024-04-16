#!/bin/bash

# Install required packages
pip install -r requirements.txt

# Download spike trains for TriDy
cd ./TriDy
curl -O https://zenodo.org/record/4290212/files/input_data.zip
unzip input_data.zip 
for f in $(ls input_data/); do mv ${f} data; done

# Extract data 
cd data && python extract_data.py
