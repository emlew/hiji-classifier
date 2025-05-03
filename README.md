# Hiji Classifier

This repository holds our collected data and work with many analysis tools.

## The Goal

We want to predict header impacts based on collected data

## Contents

### Data

This folder has a subfolder `raw` that contains `.csv` files containing timestamp and acceleration/rate data. The filenames represent the date, followed by a .[num] if there were multiple files from the same day. Walking-only files are prepended with `'W-'`. The `impacts.txt` file contains timestamps for each recorded impact, as observed in video data. It is formatted to be read by `tools/processing.py`.

### Results

Here we store visualizations generated from scripts. These figures are out of context. Refer to the project's Google Drive for reports with more context.

### Scripts

In this folder are independent scripts to analyze and visualize the data. Some key scripts include:

- `kalman_filter.py` uses a Kalman filtering technique to reduce noise
- `model_comparison.py` runs many models and compares their effectiveness
- `variable_analysis.py` identifies how different variables contribute to the model

Other scripts are named after certain models--these scripts attempt to find optimal parameters for their namesake model.

### Tools

This folder contains helper functions used in other scripts. The most important of these is `processing.py`, which generates the `final_preprocessed_data.csv` that is used in all analysis. 
