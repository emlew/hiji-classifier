import pandas as pd
import numpy as np
import glob  

data_folder = "./data/*.csv"
bin_size = "1.5S"
output_files = []  # Store intermediate files

# List of impact timestamps
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

impact_times = pd.to_datetime(datetimes)

# Cut out extraneous timestamps from certain files -- combats sensor behavior
def shorten_df(file, df):
    if file == './data/W-2-28.csv':
        return df[df['isoTimestamp'].between('2025-02-28 23:24:00.000000+00:00', '2025-02-28 23:27:30.000000+00:00')]
    if file == './data/2-21.6.csv':
        return df[df['isoTimestamp'].between('2025-02-21 19:50:00.000000+00:00', '2025-02-21 19:50:30.000000+00:00')]
    if file == './data/2-21.8.csv':
        return df[df['isoTimestamp'].between('2025-02-21 19:50:00.000000+00:00', '2025-02-21 19:50:30.000000+00:00')]
    if file == './data/2-21.10.csv':
        return df[df['isoTimestamp'].between('2025-02-21 20:04:30.000000+00:00', '2025-02-21 20:05:00.000000+00:00')]
    if file == './data/3-11.csv':
        return df[df['isoTimestamp'].between('2025-03-10 10:19:25.422000+00:00','2025-03-12 21:19:25.422000+00:00')]
    return df

# Process each file separately
for i, file in enumerate(glob.glob(data_folder)):
    print(f"Processing {file}...")
    
    df = pd.read_csv(file, dtype={"accelX": np.float32, "accelY": np.float32, "accelZ": np.float32, 
                              "rateX": np.float32, "rateY": np.float32, "rateZ": np.float32})
    df["isoTimestamp"] = pd.to_datetime(df["isoTimestamp"], utc=True) 
    df = shorten_df(file, df) 
    df = df.set_index("isoTimestamp").resample(bin_size).agg(["mean", "var", "max", "min"])  

    # Flatten column names
    df.columns = ["_".join(col) for col in df.columns]
    df.reset_index(inplace=True)

    # Convert timestamps to numeric format for efficient lookup
    df["timestamp_num"] = df["isoTimestamp"].astype("int64")  # Nanoseconds since epoch
    impact_nums = impact_times.astype("int64")  # Convert impact times too

    # Use searchsorted to efficiently check impact range
    df["impact"] = 0  # Default to 0
    idxs = np.searchsorted(df["timestamp_num"].values, impact_nums)

    # Ensure we stay in bounds and mark impact bins
    for idx in idxs:
        if idx < len(df):
            df.at[idx, "impact"] = 1  # Mark only relevant rows

    # Drop the temporary numeric timestamp column
    df.drop(columns=["timestamp_num"], inplace=True)

    # Save each file separately
    output_file = f"intermediate/processed_part_{i}.csv"
    df.to_csv(output_file, index=False)
    output_files.append(output_file)

print("Processing complete! Now merging files...")

# Merge all processed files
df_final = pd.concat([pd.read_csv(f) for f in output_files])
df_final.to_csv("./final_preprocessed_data.csv", index=False)

print("Final dataset saved as final_preprocessed_data.csv.")
