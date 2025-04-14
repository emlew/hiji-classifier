import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pykalman import KalmanFilter

df = pd.read_csv('data/Headers test 2-3-11.csv')

df['isoTimestamp'] = pd.to_datetime(df['isoTimestamp'])
df = df.sort_values(by='isoTimestamp')
df = df[df['isoTimestamp'].dt.date == pd.to_datetime('2025-03-11').date()]

# state transition matrix
dt = 0.01  # 10ms
F = np.array([
    [1, dt, 0.5*dt**2, 0, 0, 0],
    [0, 1, dt, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, dt, 0.5*dt**2],
    [0, 0, 0, 0, 1, dt],
    [0, 0, 0, 0, 0, 1]
])

# observation matrix
H = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0]
])

# initial state
initial_state = np.zeros(6)

# initial covariance (uncertainty)
initial_covariance = np.eye(6) * 0.1

# process noise (how much we expect state to vary)
Q = np.eye(6) * 0.01

# measurement noise
R = np.diag([0.1, 0.1, 0.05, 0.05])

kf = KalmanFilter(
    transition_matrices=F,
    observation_matrices=H,
    initial_state_mean=initial_state,
    initial_state_covariance=initial_covariance,
    transition_covariance=Q,
    observation_covariance=R
)

observations = df[['accelX', 'accelY', 'rateX', 'rateY']].values
timestamps = pd.to_datetime(df['isoTimestamp']).values

filtered_state_means, filtered_state_covariances = kf.filter(observations)

# calculate innovation (difference between predicted and actual)
predicted_observations = np.dot(H, filtered_state_means.T).T
innovations = observations - predicted_observations

# calculate innovation covariance
S = np.zeros((len(observations), H.shape[0], H.shape[0]))
for i in range(len(observations)):
    S[i] = np.dot(H, np.dot(filtered_state_covariances[i], H.T)) + R

# calculate mahalanobis distance for anomaly detection
mahalanobis_dist = np.zeros(len(observations))
for i in range(len(observations)):
    mahalanobis_dist[i] = np.sqrt(np.dot(innovations[i], np.dot(np.linalg.inv(S[i]), innovations[i])))

# detect impacts
impact_threshold = 500
potential_impacts = mahalanobis_dist > impact_threshold

# plt.figure(figsize=(12, 6))
plt.subplot(3,1,1)
plt.plot(timestamps, mahalanobis_dist, label='Mahalanobis Distance')
# plt.scatter(timestamps[potential_impacts], mahalanobis_dist[potential_impacts], 
#             color='red', label='Potential Impacts')
plt.title('Impact Detection Using Kalman Filter')
plt.xlabel('Time')
plt.ylabel('Mahalanobis Distance')
plt.tick_params('x', labelbottom=False)
# plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.subplot(3,1,2)
plt.plot(df['isoTimestamp'], df['accelX'], label='X-axis')
plt.plot(df['isoTimestamp'], df['accelY'], label='Y-axis')
plt.plot(df['isoTimestamp'], df['accelZ'], label='Z-axis')
plt.xlabel('Time')
plt.ylabel('Acceleration (g)')
plt.tick_params('x', labelbottom=False)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.subplot(3,1,3)
plt.plot(df['isoTimestamp'], df['rateX'], label='X-axis')
plt.plot(df['isoTimestamp'], df['rateY'], label='Y-axis')
plt.plot(df['isoTimestamp'], df['rateZ'], label='Z-axis')
plt.xlabel('Time')
plt.ylabel('Rotation Rate (deg/s)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

import time

start_time = time.time()

result = 1
for i in range(1, 100001):
  result *= i

end_time = time.time()

execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")