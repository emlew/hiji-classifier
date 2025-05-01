import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

"""
This script generates an animation of acceleration numbers with a running timestamp. It can be played alongside live footage.
"""

# Load and prepare data
df = pd.read_csv('./data/raw/2-21.6.csv')
df["isoTimestamp"] = pd.to_datetime(df["isoTimestamp"], utc=True)
df = df[df['isoTimestamp'].between('2025-02-21 19:50:05.000000+00:00', '2025-02-21 19:50:30.000000+00:00')]

# Elapsed time in seconds from the start
df['elapsed'] = (df['isoTimestamp'] - df['isoTimestamp'].iloc[0]).dt.total_seconds()

# Setup Figure and Subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

window = 3  # window in seconds

# Set y-limits
ax1.set_ylim(df[['accelX', 'accelY', 'accelZ']].min().min() - 1,
             df[['accelX', 'accelY', 'accelZ']].max().max() + 1)
ax2.set_ylim(df[['rateX', 'rateY', 'rateZ']].min().min() - 1,
             df[['rateX', 'rateY', 'rateZ']].max().max() + 1)

# Plot lines
line_ax, = ax1.plot([], [], label='accelX', color='r')
line_ay, = ax1.plot([], [], label='accelY', color='g')
line_az, = ax1.plot([], [], label='accelZ', color='b')
ax1.set_ylabel("Acceleration")
ax1.legend(loc="upper left")

line_rx, = ax2.plot([], [], label='rateX', color='r')
line_ry, = ax2.plot([], [], label='rateY', color='g')
line_rz, = ax2.plot([], [], label='rateZ', color='b')
ax2.set_ylabel("Rate")
ax2.set_xlabel("Time (s)")
ax2.legend(loc="upper left")

# Time display
time_text = fig.text(0.5, 0.94, '', ha='center', va='bottom', fontsize=12, fontweight='bold')

def init():
    for line in [line_ax, line_ay, line_az, line_rx, line_ry, line_rz]:
        line.set_data([], [])
    ax1.set_xlim(0, window)
    time_text.set_text('')
    return line_ax, line_ay, line_az, line_rx, line_ry, line_rz, time_text

def update(frame):
    current_time = df['elapsed'].iloc[frame]
    start_time = max(0, current_time - window)
    
    visible = df[df['elapsed'].between(start_time, current_time)]
    xdata = visible['elapsed']
    
    line_ax.set_data(xdata, visible['accelX'])
    line_ay.set_data(xdata, visible['accelY'])
    line_az.set_data(xdata, visible['accelZ'])

    line_rx.set_data(xdata, visible['rateX'])
    line_ry.set_data(xdata, visible['rateY'])
    line_rz.set_data(xdata, visible['rateZ'])

    ax1.set_xlim(start_time, current_time)
    ax2.set_xlim(start_time, current_time)

    time_text.set_text(f"Time: {df['isoTimestamp'].iloc[frame].strftime('%H:%M:%S.%f')[:-3]}")
    
    return line_ax, line_ay, line_az, line_rx, line_ry, line_rz, time_text

ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(df),
    init_func=init,
    interval=20,  # milliseconds between frames, optional fine-tune
    repeat=False,
)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


ani.save('accel_visualization.mp4', writer='ffmpeg', fps=30)
