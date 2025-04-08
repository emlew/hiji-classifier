import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy
from tools.cleanup import enforce_file_bounds, get_impacts

impacts = get_impacts()

df = pd.read_csv('./data/3-2.csv', dtype={"accelX": np.float32, "accelY": np.float32, "accelZ": np.float32, 
                            "rateX": np.float32, "rateY": np.float32, "rateZ": np.float32})
df["isoTimestamp"] = pd.to_datetime(df["isoTimestamp"], utc=True)
df = enforce_file_bounds('./data/3-2.csv', df) 

df['resLA'] = np.abs(np.sqrt(df['rateX']**2+df['rateY']**2+df['rateZ']**2)) # resultant linear acceleration
df['resAA'] = np.abs(np.sqrt(df['accelX']**2+df['accelY']**2+df['accelZ']**2)) # resultant angular acceleration

# HIC15 = max[ (t2 - t1) * ( (1/(t2-t1)) * integral(a(t) dt, t1, t2) )^2.5 ]
df.sort_values('isoTimestamp')
df['timestamp'] = df['timestamp']*1000
df['timestamp'] = df['timestamp'].astype(int)
delta = 15
ms_time_range = np.arange(np.min(df['timestamp']),np.max(df['timestamp'])-delta)
hicsLA = [0]*len(ms_time_range)
hicsAA = [0]*len(ms_time_range)

for i in range(len(ms_time_range)):
    t1 = ms_time_range[i]
    t2 = t1 + delta
    vals = df[df['timestamp'].between(t1, t2)]
    if len(vals) == 0:
        continue
    hicsLA[i] = (delta) * ((1/delta)*scipy.integrate.simpson(y=vals['resLA'],x=vals['timestamp']))**2.5    
    hicsAA[i] = (delta) * ((1/delta)*scipy.integrate.simpson(y=vals['resAA'],x=vals['timestamp']))**2.5
    
# print(hicsLA[:10])
# print(hicsAA[:10])
fig, axs = plt.subplots(2,1)
axs[0].plot(hicsAA)
axs[1].plot(hicsLA)
plt.show()