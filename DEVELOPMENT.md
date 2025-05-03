# Developer Guide

## Requirements

This repo requires a valid [Python](https://www.python.org/downloads/) distribution. You can check this with the command `python3 -version` on Mac or `py -version` on Windows.

You will also need the following Python packages:

- numpy
- pandas
- matplotlib
- xgboost
- sklearn
- statsmodels
- seaborn
- scipy
- pykalman

Packages can be installed with the command `python3 -m pip install [package name]` on Mac, or `py -m pip install [package name]` on Windows.

Developing in VSCode is recommended. VSCode will flag import statements for missing packages.

## Troubleshooting

If you encounter runtime errors such as:

```
ModuleNotFoundError: No module named 'tools'
```

You may need to add to the filepath by adding this script to the beginning of the file:

```
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
```

## Running Locally and Testing

All scripts can be run independently using the Run Python File command in the VSCode IDE. Code can be debugged using the inbuilt Python Debugger.

Before running `processing.py`, you'll need to create an empty `intermediate` folder in the `data` module. This is an untracked folder that holds intermediate files during processing. It can be emptied after each run, but the folder must exist or `processing.py` will fail.

## Adding Data

Data can be added  to the `data/raw` folder, with a name following the naming conventions specified in the README. When adding data, make sure to visualize it first and add to the `enforce_file_bounds` function as needed.

This is necessary because our sensors often have more time recorded than intended. For instance, the data from March 11 has a minimum timestamp of `2025-02-26 00:24:28.870000+00:00` and a maximum of `2106-02-07 06:28:21.554000+00:00`. We don't need (and don't want) the extraneous data, so we add a condition to trim the data to the timeframe in which we were recording.

## Timestamps

Sensors record ISO timestamps in UTC time and Unix timestamps. [This site](https://www.timestamp-converter.com/) is helpful for manual conversion if needed.

## Code Style

Code should follow Python formatting best practices. All code should be readable and thoroughly commented.