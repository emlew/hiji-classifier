import numpy as np
import pandas as pd

def get_impacts():
    """
    Reads and returns impact list from impacts.txt in data folder.
    
    Returns:
    - array of DateTimeIndex objects corresponding to each impact
    """
    datetimes = np.array([])
    with open('./data/impacts.txt', 'r') as file:
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