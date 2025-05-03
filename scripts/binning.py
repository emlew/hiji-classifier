from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from tools.processing import get_impacts, get_interval_binned_data, iqr

"""
This script generates two examples of how binning works. Each example displays unbinned data on the left and binned data on the right.
"""

# ---------------- Example 1 ----------------
# read in 3-2 data and impacts
impacts = get_impacts()
unbinned = pd.read_csv('./data/raw/3-2.csv')
unbinned["isoTimestamp"] = pd.to_datetime(unbinned["isoTimestamp"], utc=True)
unbinned = unbinned[unbinned['isoTimestamp'].between('2025-03-03 00:00:00.000000+00:00', '2025-03-03 04:20:00.000000+00:00')]
unbinned['resLA'] = np.abs(np.sqrt(unbinned['rateX']**2+unbinned['rateY']**2+unbinned['rateZ']**2))

# generate binned data according to these params
data_folder = "./data/raw/*.csv"
bin_size = "1.5s"
aggregates = ["mean", "var", "max", "min"]
get_interval_binned_data(data_folder,bin_size,aggregates,False)

# read in binned data, trim to 3-2 only
binned = pd.read_csv('./final_preprocessed_data.csv')
binned["isoTimestamp"] = pd.to_datetime(binned["isoTimestamp"], utc=True, format='mixed')
binned = binned[binned['isoTimestamp'].between('2025-03-03 00:00:00.000000+00:00', '2025-03-03 04:20:00.000000+00:00')]

# plot with binned data on left and unbinned on the right
_, (ax1, ax2) = plt.subplots(1,2)
ax1.plot(unbinned['isoTimestamp'],unbinned['rateX'])
ax2.plot(binned['isoTimestamp'],binned['rateX_mean'])

ax1.set_ylim([-500,500])
ax2.set_ylim([-500,500])

ax1.set_title("Raw data for rate X")
ax2.set_title("Binned data at 1.5s intervals for mean of rate X")

plt.show()

# ---------------- Example 2 ----------------
# generate binned data according to these params
data_folder = "./data/raw/*.csv"
bin_size = "1.0s"
aggregates = ["median", iqr]
get_interval_binned_data(data_folder,bin_size,aggregates,True)

# read in binned data, trim to 3-2 only
binned = pd.read_csv('./final_preprocessed_data.csv')
binned["isoTimestamp"] = pd.to_datetime(binned["isoTimestamp"], utc=True, format='mixed')
binned = binned[binned['isoTimestamp'].between('2025-03-03 00:00:00.000000+00:00', '2025-03-03 04:20:00.000000+00:00')]

# plot with binned data on left and unbinned on the right
_, (ax1, ax2) = plt.subplots(1,2)
ax1.plot(unbinned['isoTimestamp'],unbinned['resLA'])
ax2.plot(binned['isoTimestamp'],binned['resLA_median'])

ax1.set_title("Raw data for resultant linear acceleration")
ax2.set_title("Binned data at 1.0s intervals for median of resultant linear acceleration")

plt.show()
