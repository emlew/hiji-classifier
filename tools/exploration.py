from matplotlib import pyplot as plt
import pandas as pd

from lineplot_tools import plot_vs_time

# df = pd.read_csv('./final_preprocessed_data.csv')
# plt.plot(df['isoTimestamp'], df['impact'])
# plt.show()

df = pd.read_csv('./data/W-3-11.1.csv')
df['isoTimestamp'] = pd.to_datetime(df['isoTimestamp'])
# df = df[df['isoTimestamp'].between('2025-03-10 10:19:25.422000+00:00','2025-03-12 21:19:25.422000+00:00')]

print(max(df['isoTimestamp']))
print(min(df['isoTimestamp']))
print(len(df['isoTimestamp']))

plot_vs_time(df,'accel','Accels')
plot_vs_time(df,'rate','Accels')