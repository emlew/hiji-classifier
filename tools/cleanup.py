import numpy as np
import pandas as pd

def get_impacts():
    datetimes = np.array([])
    with open('./impacts.txt', 'r') as file:
        for line in file:
            line = line.strip()
            if line:  # Ensure the line is not empty
                if line.startswith('#'):
                    continue
                try:
                    dt = pd.to_datetime(line)
                    datetimes = np.append(datetimes, dt)
                except ValueError:
                    print(f"Invalid datetime format: {line}. Skipping.")

    return pd.to_datetime(datetimes)

# Cut out extraneous timestamps from certain files -- combats sensor behavior
def enforce_file_bounds(file, df):
    if file == './data/raw/W-2-28.csv':
        return df[df['isoTimestamp'].between('2025-02-28 23:24:00.000000+00:00', '2025-02-28 23:27:30.000000+00:00')]
    if file == './data/raw/2-21.6.csv':
        return df[df['isoTimestamp'].between('2025-02-21 19:50:00.000000+00:00', '2025-02-21 19:50:30.000000+00:00')]
    if file == './data/raw/2-21.8.csv':
        return df[df['isoTimestamp'].between('2025-02-21 19:50:00.000000+00:00', '2025-02-21 19:50:30.000000+00:00')]
    if file == './data/raw/2-21.10.csv':
        return df[df['isoTimestamp'].between('2025-02-21 20:04:30.000000+00:00', '2025-02-21 20:05:00.000000+00:00')]
    if file == './data/raw/3-11.csv':
        return df[df['isoTimestamp'].between('2025-03-10 10:19:25.422000+00:00','2025-03-12 21:19:25.422000+00:00')]
    return df